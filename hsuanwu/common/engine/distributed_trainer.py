from collections import deque

import hydra
import numpy as np
import torch
import threading
import traceback
from torch import multiprocessing as mp

from hsuanwu.common.engine import BasePolicyTrainer, utils
from hsuanwu.common.logger import *
from hsuanwu.common.typing import DictConfig, Env, Logger


from hsuanwu.xploit.learner import IMPALALearner
from hsuanwu.xploit.storage import DistributedStorage

class DistributedTrainer(BasePolicyTrainer):
    """Trainer for on-policy algorithms.

    Args:
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.
        cfgs (DictConfig): Dict config for configuring RL algorithms.

    Returns:
        On-policy trainer instance.
    """
    def __init__(self, train_env: Env, test_env: Env, cfgs: DictConfig) -> None:
        super().__init__(train_env, test_env, cfgs)

        self._learner = IMPALALearner(
            observation_space=None,
            action_space=None,
            action_type='dis',
            device='cuda',
            feature_dim=1024,
            lr=1e-4
        )

        # create shared storages
        self._shared_storages = DistributedStorage.create_storages(
            obs_shape=self._train_env.observation_space.shape,
            action_shape=self._train_env.action_space.n,
            action_type='dis',
            num_steps=self._cfgs.num_steps,
            num_storages=self._cfgs.num_storages
        )


    def train(self) -> None:
        def sample_and_udpate(thread_idx, lock=threading.Lock()) -> None:
            """Thread target for the learning process."""
            steps = 0
            metrics = None

            while steps < self._cfgs.num_train_steps:
                batch, actor_state = DistributedStorage.sample(
                    free_queue=free_queue,
                    full_queue=full_queue,
                    storages=self._storages,
                    init_actor_state_storages=init_actor_state_storages,
                    cfgs=self._cfgs,
                )
                metrics = self._learner.update(actor=self._learner.actor,
                                               learner=self._learner.learner,
                                               batch=batch,
                                               actor_state=actor_state,
                                               opt=self._learner.opt,
                                               scheduler=self._learner.lr_scheduler,
                                               cfgs=self._cfgs
                                               )
                
                with lock:
                    step += self._cfgs.num_steps * self._cfgs.storage.batch_size
            
            if thread_idx == 0:
                self._logger.log(INFO, 'Sample and udpate: ')
        #####################################################
        init_actor_state_storages = list()
        for _ in range(self._cfgs.num_storages):
            state = self._learner.actor.initial_state(batch_size=1)
            for t in state:
                t.share_memory_()
            init_actor_state_storages.append(state)
        
        # actor pool
        actor_pool = list()
        ctx = mp.get_context('fork')
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()

        for i in range(self._cfgs.num_actors):
            actor = ctx.Process(
                target=self.act,
                args=(i, 
                      free_queue,
                      full_queue,
                      self._train_env[i], 
                      self._learner.actor, 
                      self._shared_storages,
                      init_actor_state_storages,
                      self._cfgs,
                      self._logger)
            )
            actor.start()
            actor_pool.append(actor)
        
        # Serialize the data before acquiring the lock
        for i in range(self._cfgs.num_storages):
            free_queue.put(i)
        
        # set threads for multiple learners
        threads = list()
        for i in range(self._cfgs.num_threads):
            thread = threading.Thread(
            target=sample_and_udpate, name='batch-and-learn-%d' % i, args=(i,))
            thread.start()
            threads.append(thread)

    @staticmethod
    def act(actor_idx: int,
            free_queue: SimpleQueue,
            full_queue: SimpleQueue,
            env: Env, 
            actor: NNModule, 
            storages: List[Storage], 
            init_actor_state_storages: List,
            cfgs: DictConfig,
            logger: Logger
            ) -> None:
        """Running loop for single actor.

        Args:
            actor_idx (int): The index of idx.
            free_queue (SimpleQueue): .
            full_queue (SimpleQueue): .
            env (Env): A single environment.
            actor (NNModule): The actor network.
            storages (List[Storage]): A list of Hsuanwu 'DistributedStorage' instances.
            init_actor_state_storages (List): A list for storing the initial states of the actor.
            cfgs (DictConfig): Dict config.
            logger (Logger): The Hsuanwu 'logger' instance.
        
        Returns:
            None.
        """
        try:
            logger.log(INFO, f'Actor {actor_idx} started!')

            env_state = env.reset()
            actor_state = actor.initial_state(batch_size=1)
            actor_output, _ = actor(env_state, actor_state)

            while True:
                idx = free_queue.get()
                if idx is None:
                    break
                
                 # Write old rollout end.
                for key in env_state:
                    storages[key][idx][0, ...] = env_state[key]
                for key in actor_output:
                    storages[key][idx][0, ...] = actor_output[key]
                for actor_idx, tensor in enumerate(actor_state):
                    init_actor_state_storages[idx][actor_idx][...] = tensor

                # Do new rollout
                for t in range(cfgs.num_steps):
                    with torch.no_grad():
                        actor_output, actor_state = actor(env_output, actor_state)

                    env_output = env.step(actor_output['action'])

                    for key in env_output:
                        storages[key][idx][t + 1, ...] = env_output[key]
        
                    for key in actor_output:
                        storages[key][idx][t + 1, ...] = actor_output[key]

                full_queue.put(idx)

            if actor_idx == 0:
                logger.log(INFO, f'Actor {actor_idx} time cost: ')

        except KeyboardInterrupt:
            pass  
        except Exception as e:
            logger.log(ERROR, f'Exception in worker process {actor_idx}')
            traceback.print_exc()
            raise e
