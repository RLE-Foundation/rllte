import os

os.environ["OMP_NUM_THREADS"] = "1"
import pprint
import threading
import time
import timeit
import traceback

import numpy as np
import hydra
import torch
from torch import multiprocessing as mp

from hsuanwu.common.engine import BasePolicyTrainer
from hsuanwu.common.logger import Logger, INFO, ERROR
from hsuanwu.common.typing import DictConfig, Env, NNModule, List, Storage, Tensor, Dict, Env, Ndarray

class Environment:
    """An env wrapper to adapt to the distributed trainer.

    Args:
        env (env): A Gym-like env.

    Returns:
        Processed env.
    """
    def __init__(self, env: Env) -> None:
        self.env = env
        self.episode_return = None
        self.episode_step = None
    
    def reset(self, seed) -> Dict[str, Tensor]:
        """Reset the environment.
        """
        init_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        init_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        init_terminated = torch.ones(1, 1, dtype=torch.uint8)
        init_truncated = torch.ones(1, 1, dtype=torch.uint8)

        obs, info = self.env.reset(seed=seed)
        obs = self._format_obs(obs)

        return dict(
            obs=obs,
            reward=init_reward,
            terminated=init_terminated,
            truncated=init_truncated,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=init_last_action,
        )
    
    def step(self, action: Tensor) -> Dict[str, Tensor]:
        """Step function that returns a dict consists of current and history observation and action.

        Args:
            action (Tensor): Action tensor.

        Returns:
            Step information dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if terminated or truncated:
            obs, info = self.env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        obs = self._format_obs(obs)
        reward = torch.as_tensor(reward, dtype=torch.float32).view(1, 1)
        terminated = torch.as_tensor(terminated, dtype=torch.uint8).view(1, 1)
        truncated = torch.as_tensor(truncated, dtype=torch.uint8).view(1, 1)

        return dict(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )
    
    def close(self) -> None:
        """Close the environment.
        """
        self.env.close()

    def _format_obs(self, obs: Ndarray) -> Tensor:
        """Reformat the observation by adding an time dimension.

        Args:
            obs (NdArray): Observation.

        Returns:
            Formatted observation.
        """
        obs = torch.from_numpy(np.array(obs))
        return obs.view((1, 1) + obs.shape)
        

class DistributedTrainer(BasePolicyTrainer):
    """Trainer for on-policy algorithms.

    Args:
        train_env (Env): A list of Gym-like environments for training.
        test_env (Env): A Gym-like environment for testing.
        cfgs (DictConfig): Dict config for configuring RL algorithms.

    Returns:
        On-policy trainer instance.
    """

    def __init__(self, cfgs: DictConfig, train_env: Env, test_env: Env = None) -> None:
        super().__init__(cfgs, train_env, test_env)
        self._logger.log(INFO, f"Deploying DistributedTrainer...")
        # xploit part
        self._learner = hydra.utils.instantiate(self._cfgs.learner)
        # self._learner.actor.encoder = hydra.utils.instantiate(self._cfgs.encoder)
        # self._learner.learner.encoder = hydra.utils.instantiate(self._cfgs.encoder)
        # self._learner.learner.to(self._device)
        # self._learner.opt = torch.optim.RMSprop(
        #     self._learner.learner.parameters(), lr=self._learner.lr, eps=self._learner.eps
        # )
        # def lr_lambda(epoch):
        #     return 1. - min(epoch * self._cfgs.num_steps * self._cfgs.num_learners, 
        #                     self._cfgs.num_train_steps) / self._cfgs.num_train_steps
        
        # self._learner.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self._learner.opt, lr_lambda)
        ## TODO: build storage
        self._shared_storages = hydra.utils.instantiate(self._cfgs.storage)
        self._train_env = self._train_env.envs

        # xplore part
        ## build distribution
        # if "Noise" in self._cfgs.distribution._target_:
        #     dist = hydra.utils.instantiate(self._cfgs.distribution)
        # else:
        #     dist = hydra.utils.get_class(self._cfgs.distribution._target_)
        # self._learner.dist = dist
        # self._learner.actor.dist = dist
        # self._learner.learner.dist = dist

    @staticmethod
    def act(
        cfgs: DictConfig,
        logger: Logger,
        actor_idx: int,
        gym_env: Env,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        actor_model: NNModule,
        storages: List[Storage],
        init_actor_states: List[Tensor],
    ) -> None:
        """Sampling function for each actor. 

        Args:
            cfgs (DictConfig): Training configs.
            logger (Logger): Hsuanwu logger.
            actor_idx (int): The index of actor.
            gym_env (Env): A Gym-like environment.
            free_queue (Queue): Free queue for communication.
            full_queue (Queue): Full queue for communication.
            actor_model (NNMoudle): Actor network.
            storages (List[Storage]): A list of shared storages.
            init_actor_states: (List[Tensor]): Initial states for LSTM.

        Returns:
            None.
        """
        try:
            logger.log(INFO, f"Actor {actor_idx} started!")
            
            env = Environment(gym_env)
            seed = actor_idx * int.from_bytes(os.urandom(4), byteorder="little")
            env_output = env.reset(seed)
            actor_state = actor_model.init_state(batch_size=1)
            actor_output, _ = actor_model.get_action(env_output, actor_state)
            while True:
                idx = free_queue.get()
                if idx is None:
                    break

                # Write old rollout end.
                for key in env_output:
                    storages[key][idx][0, ...] = env_output[key]
                for key in actor_output:
                    storages[key][idx][0, ...] = actor_output[key]
                for i, tensor in enumerate(actor_state):
                    init_actor_states[idx][i][...] = tensor

                # Do new rollout.
                for t in range(cfgs.num_steps):
                    with torch.no_grad():
                        actor_output, actor_state = actor_model.get_action(env_output, actor_state)
                    env_output = env.step(actor_output["action"])

                    for key in env_output:
                        storages[key][idx][t + 1, ...] = env_output[key]
                    for key in actor_output:
                        storages[key][idx][t + 1, ...] = actor_output[key]

                full_queue.put(idx)

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            logger.log(ERROR, f"Exception in worker process {actor_idx}!")
            traceback.print_exc()
            raise e

    def train(self):
        """Training function
        """
        global_step = 0
        metrics = dict()

        def sample_and_update(i, lock=threading.Lock()):
            """Thread target for the learning process."""
            nonlocal global_step, metrics
            while self._global_step < self._cfgs.num_train_steps:
                batch, actor_state = self._shared_storages.sample(
                    device=self._cfgs.device,
                    batch_size=self._cfgs.storage.batch_size,
                    free_queue=free_queue,
                    full_queue=full_queue,
                    storages=self._shared_storages.storages,
                    init_actor_states=init_actor_states,
                )
                metrics = self._learner.update(
                    cfgs=self._cfgs,
                    actor_model=self._learner.actor,
                    learner_model=self._learner.learner,
                    batch=batch,
                    init_actor_state=actor_state,
                    optimizer=self._learner.opt,
                    lr_scheduler=self._learner.lr_scheduler,
                )
                with lock:
                    global_step += self._cfgs.num_steps * self._cfgs.storage.batch_size
                    # self._global_step += self._cfgs.num_steps * self._cfgs.storage.batch_size

        # TODO: Add initial RNN state.
        init_actor_states = []
        for _ in range(self._cfgs.storage.num_storages):
            state = self._learner.actor.init_state(batch_size=1)
            for t in state:
                t.share_memory_()
            init_actor_states.append(state)

        actor_pool = []
        ctx = mp.get_context("fork")
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()

        for actor_idx in range(self._cfgs.num_actors):
            actor = ctx.Process(
                target=self.act,
                args=(
                    self._cfgs,
                    self._logger,
                    actor_idx,
                    self._train_env[actor_idx],
                    free_queue,
                    full_queue,
                    self._learner.actor,
                    self._shared_storages.storages,
                    init_actor_states,
                ),
            )
            actor.start()
            actor_pool.append(actor)

        for m in range(self._cfgs.storage.num_storages):
            free_queue.put(m)

        threads = []
        for i in range(self._cfgs.num_learners):
            thread = threading.Thread(
                target=sample_and_update, name="sample-and-update-%d" % i, args=(i,)
            )
            thread.start()
            self._logger.log(INFO, f"Learner {i} started!")
            threads.append(thread)

        timer = timeit.default_timer
        try:
            last_checkpoint_time = timer()
            while global_step < self._cfgs.num_train_steps:
                start_step = global_step
                start_time = timer()
                time.sleep(5)

                if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                    last_checkpoint_time = timer()

                sps = (global_step - start_step) / (timer() - start_time)
                if metrics.get("episode_returns", None):
                    mean_return = (
                        "Return per episode: %.1f. " % metrics["mean_episode_return"]
                    )
                else:
                    mean_return = ""
                total_loss = metrics.get("total_loss", float("inf"))
                self._logger.log(
                    INFO,
                    f"Steps {global_step} @ {sps} SPS. Loss {total_loss}. {mean_return}Stats:\n{pprint.pformat(metrics)}",
                )
        except KeyboardInterrupt:
            # TODO: join actors then quit.
            return
        else:
            for thread in threads:
                thread.join()
            self._logger.log(INFO, "Training Accomplished!")
        finally:
            for _ in range(self._cfgs.num_actors):
                free_queue.put(None)
            for actor in actor_pool:
                actor.join(timeout=1)

    def test(self) -> None:
        """Testing function."""
        pass

    def save(self) -> None:
        pass
