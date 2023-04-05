import pprint
import random
import threading
import time
import traceback
from collections import deque

import hydra
import numpy as np
import torch
from torch import multiprocessing as mp

from hsuanwu.common.engine import utils
from hsuanwu.common.logger import *
from hsuanwu.common.timer import Timer
from hsuanwu.common.typing import DictConfig, Env
from hsuanwu.xploit.learner import IMPALALearner
from hsuanwu.xploit.learner.impala import AtariNet
from hsuanwu.xploit.storage import DistributedStorage

import os
os.environ["OMP_NUM_THREADS"] = "1" 

def create_env():
    return utils.wrap_pytorch(
        utils.wrap_deepmind(
            utils.make_atari('PongNoFrameskip-v4'),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )

import logging
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

def beast_act(
    cfgs,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers,
    initial_agent_state_buffers,
):
    try:
        print("Actor %i started.", actor_index)
        timings = utils.Timings()  # Keep track of how fast things are.

        gym_env = create_env()
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = utils.Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        print('runned here 111')
        agent_output, unused_state = model(env_output, agent_state)
        print('runned here')
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(cfgs.num_steps):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

def train(cfgs):  # pylint: disable=too-many-branches, too-many-statements
    T = cfgs.num_steps
    B = cfgs.storage.batch_size

    env = create_env()

    model = AtariNet(observation_shape=(4, 84, 84), num_actions=18)
    buffers = DistributedStorage.create_storages(
            obs_shape=(4, 84, 84),
            action_shape=1,
            action_type='dis',
            num_steps=cfgs.num_steps,
            num_storages=cfgs.storage.num_storages
        )

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(cfgs.storage.num_storages):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(cfgs.num_actors):
        actor = ctx.Process(
            target=beast_act,
            args=(
                cfgs,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

class DistributedTrainer:
    """Trainer for on-policy algorithms.

    Args:
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.
        cfgs (DictConfig): Dict config for configuring RL algorithms.

    Returns:
        On-policy trainer instance.
    """
    def __init__(self, train_env: Env, test_env: Env, cfgs: DictConfig) -> None:
        pass
#         self._cfgs = cfgs
#         self._work_dir = Path.cwd()
#         self._logger = Logger(log_dir=self._work_dir)
#         self._timer = Timer()
#         self._device = torch.device(cfgs.device)
#         # set seed
#         self._seed = cfgs.seed
#         torch.manual_seed(seed=cfgs.seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(cfgs.seed)
#         np.random.seed(cfgs.seed)
#         random.seed(cfgs.seed)
#         # debug
#         self._logger.log(INFO, "Invoking Hsuanwu Engine...")

#         self._learner = IMPALALearner(
#             observation_space=None,
#             action_space=None,
#             action_type='dis',
#             device='cuda',
#             feature_dim=1024,
#             lr=self._cfgs.learner.lr,
#             eps=self._cfgs.learner.eps
#         )

#         # create shared storages
#         self._shared_storages = DistributedStorage.create_storages(
#             obs_shape=(4, 84, 84),
#             action_shape=1,
#             action_type='dis',
#             num_steps=self._cfgs.num_steps,
#             num_storages=self._cfgs.storage.num_storages
#         )


#     def train(self) -> None:
#         step, stats = 0, {}
#         #####################################################
#         def sample_and_udpate(thread_idx, lock=threading.Lock()) -> None:
#             """Thread target for the learning process."""
#             nonlocal step, stats
#             print(thread_idx, 'updating!')
#             while step < self._cfgs.num_train_steps:
#                 print(step, self._cfgs.num_train_steps, 'flag 111')
#                 batch, actor_state = DistributedStorage.sample(
#                     free_queue=free_queue,
#                     full_queue=full_queue,
#                     storages=self._shared_storages,
#                     init_actor_state_storages=init_actor_state_storages,
#                     cfgs=self._cfgs)
#                 print(step, self._cfgs.num_train_steps, 'flag 222')
#                 stats = self._learner.update(actor=tmp_model,
#                                                learner=self._learner.learner,
#                                                batch=batch,
#                                                actor_state=actor_state,
#                                                opt=self._learner.opt,
#                                                scheduler=self._learner.lr_scheduler,
#                                                cfgs=self._cfgs)
#                 print(step, self._cfgs.num_train_steps, 'flag 333')
#                 with lock:
#                     step += self._cfgs.num_steps * self._cfgs.storage.batch_size
            
#             if thread_idx == 0:
#                 self._logger.log(INFO, 'Sample and udpate: ')
#         #####################################################

#         tmp_model = AtariNet(
#             observation_shape=(4, 84, 84),
#             num_actions=18
#         )
#         tmp_model.share_memory()

#         init_actor_state_storages = list()
#         for _ in range(self._cfgs.storage.num_storages):
#             state = tmp_model.initial_state(batch_size=1)
#             for t in state:
#                 t.share_memory_()
#             init_actor_state_storages.append(state)
        
#         # actor pool
#         actor_pool = list()
#         ctx = mp.get_context('fork')
#         free_queue = ctx.SimpleQueue()
#         full_queue = ctx.SimpleQueue()

#         for i in range(self._cfgs.num_actors):
#             actor = ctx.Process(
#                 target=beast_act,
#                 args=(i, 
#                       free_queue,
#                       full_queue,
#                     #   self._train_env[i], 
#                       tmp_model, 
#                       self._shared_storages,
#                       init_actor_state_storages,
#                       self._cfgs,)
#                     #   self._logger)
#             )
#             actor.start()
#             actor_pool.append(actor)

#         # Serialize the data before acquiring the lock
#         for i in range(self._cfgs.storage.num_storages):
#             free_queue.put(i)
        
#         # set threads for multiple learners
#         threads = list()
#         for i in range(self._cfgs.num_threads):
#             thread = threading.Thread(
#             target=sample_and_udpate, name='sample-and-update-%d' % i, args=(i,))
#             thread.start()
#             threads.append(thread)
        
#         try:
#             while step < self._cfgs.num_train_steps:
#                 start_step = step
#                 time.sleep(5)

#                 sps = (step - start_step)
#                 if stats.get("episode_returns", None):
#                     mean_return = (
#                         "Return per episode: %.1f. " % stats["mean_episode_return"]
#                     )
#                 else:
#                     mean_return = ""
                
#                 total_loss = stats.get("total_loss", float("inf"))
#                 self._logger.log(INFO, 
#                     f"Steps {step} @ {sps} SPS. Loss {total_loss} {mean_return}Stats:\n{pprint.pformat(stats)}")
#         except KeyboardInterrupt:
#             return
#         else:
#             for thread in threads:
#                 thread.join()
#             self._logger.log(INFO, "Learning finished after %d steps.", step)
#         finally:
#             for _ in range(self._cfgs.num_actors):
#                 free_queue.put(None)
#             for actor in actor_pool:
#                 actor.join(timeout=1)


#     @staticmethod
#     def act(actor_idx: int,
#             free_queue: SimpleQueue,
#             full_queue: SimpleQueue,
#             # env: Env, 
#             actor: NNModule, 
#             storages: List[Storage], 
#             init_actor_state_storages: List,
#             cfgs: DictConfig,
#             logger: Logger
#             ) -> None:
#         """Running loop for single actor.

#         Args:
#             actor_idx (int): The index of idx.
#             free_queue (SimpleQueue): .
#             full_queue (SimpleQueue): .
#             env (Env): A single environment.
#             actor (NNModule): The actor network.
#             storages (List[Storage]): A list of Hsuanwu 'DistributedStorage' instances.
#             init_actor_state_storages (List): A list for storing the initial states of the actor.
#             cfgs (DictConfig): Dict config.
#             logger (Logger): The Hsuanwu 'logger' instance.
        
#         Returns:
#             None.
#         """
#         logger.log(INFO, f'Actor {actor_idx} started!')
#         gym_env = create_env()
#         seed = actor_idx ^ int.from_bytes(os.urandom(4), byteorder="little")
#         gym_env.seed(seed)
#         env = utils.Environment(gym_env)
#         env_state = env.initial()
#         actor_state = actor.initial_state(batch_size=1)
#         print(actor_idx, 'acting 22')
#         actor_output, _ = actor(env_state, actor_state)
#         print(actor_idx, 'acting 33')

#         try:
#             logger.log(INFO, f'Actor {actor_idx} started!')

#             gym_env = create_env()
#             seed = actor_idx ^ int.from_bytes(os.urandom(4), byteorder="little")
#             gym_env.seed(seed)
#             env = utils.Environment(gym_env)
#             env_state = env.initial()
#             actor_state = actor.initial_state(batch_size=1)
#             print(actor_idx, 'acting 2')
#             actor_output, _ = actor(env_state, actor_state)
#             print(actor_idx, 'acting 3')
#             while True:
#                 idx = free_queue.get()
#                 if idx is None:
#                     break
                
#                  # Write old rollout end.
#                 for key in env_state:
#                     storages[key][idx][0, ...] = env_state[key]
#                 for key in actor_output:
#                     storages[key][idx][0, ...] = actor_output[key]
#                 for actor_idx, tensor in enumerate(actor_state):
#                     init_actor_state_storages[idx][actor_idx][...] = tensor

#                 # Do new rollout
#                 for t in range(cfgs.num_steps):
#                     with torch.no_grad():
#                         print(actor_idx, 'sampling')
#                         actor_output, actor_state = actor(env_output, actor_state)

#                     env_output = env.step(actor_output['action'])

#                     for key in env_output:
#                         storages[key][idx][t + 1, ...] = env_output[key]
        
#                     for key in actor_output:
#                         storages[key][idx][t + 1, ...] = actor_output[key]

#                 full_queue.put(idx)

#             if actor_idx == 0:
#                 logger.log(INFO, f'Actor {actor_idx} time cost: ')

#         except KeyboardInterrupt:
#             pass  
#         except Exception as e:
#             logger.log(ERROR, f'Exception in worker process {actor_idx}')
#             traceback.print_exc()
#             raise e
