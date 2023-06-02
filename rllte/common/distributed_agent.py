import os
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import hydra
import numpy as np
import omegaconf
import torch as th
from torch import multiprocessing as mp
from torch import nn

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.base_agent import BaseAgent

class Environment:
    """An env wrapper to adapt to the distributed trainer.

    Args:
        env (Env): A Gym-like env.

    Returns:
        Processed env.
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.episode_return = None
        self.episode_step = None
        if env.action_space.__class__.__name__ == "Discrete":
            self.action_type = "Discrete"
            self.action_dim = 1
        elif env.action_space.__class__.__name__ == "Box":
            self.action_type = "Box"
            self.action_dim = env.action_space.shape[0]
        else:
            raise NotImplementedError("Unsupported action type!")

    def reset(self, seed) -> Dict[str, th.Tensor]:
        """Reset the environment."""
        init_reward = th.zeros(1, 1)
        init_last_action = th.zeros(1, self.action_dim, dtype=th.int64)
        self.episode_return = th.zeros(1, 1)
        self.episode_step = th.zeros(1, 1, dtype=th.int32)
        init_terminated = th.ones(1, 1, dtype=th.uint8)
        init_truncated = th.ones(1, 1, dtype=th.uint8)

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

    def step(self, action: th.Tensor) -> Dict[str, th.Tensor]:
        """Step function that returns a dict consists of current and history observation and action.

        Args:
            action (Tensor): Action tensor.

        Returns:
            Step information dict.
        """
        if self.action_type == "Discrete":
            _action = action.item()
        elif self.action_type == "Box":
            _action = action.squeeze(0).cpu().numpy()
        else:
            raise NotImplementedError("Unsupported action type!")

        obs, reward, terminated, truncated, info = self.env.step(_action)
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if terminated or truncated:
            obs, info = self.env.reset()
            self.episode_return = th.zeros(1, 1)
            self.episode_step = th.zeros(1, 1, dtype=th.int32)

        obs = self._format_obs(obs)
        reward = th.as_tensor(reward, dtype=th.float32).view(1, 1)
        terminated = th.as_tensor(terminated, dtype=th.uint8).view(1, 1)
        truncated = th.as_tensor(truncated, dtype=th.uint8).view(1, 1)

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
        """Close the environment."""
        self.env.close()

    def _format_obs(self, obs: np.ndarray) -> th.Tensor:
        """Reformat the observation by adding an time dimension.

        Args:
            obs (NdArray): Observation.

        Returns:
            Formatted observation.
        """
        obs = th.from_numpy(np.array(obs))
        return obs.view((1, 1, *obs.shape))

class DistributedAgent(BaseAgent):
    """Trainer for distributed algorithms.

    Args:
        env (Env): A Gym-like environment for training.
        eval_env (Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_steps (int): The sample length of per rollout.
        num_actors (int): Number of actors.
        num_learners (int): Number of learners.
        num_storages (int): Number of storages.
        batch_size (int): Number of samples per batch to load.

    Returns:
        Distributed agent instance.
    """
    def __init__(self, 
                 env: gym.Env,
                 eval_env: Optional[gym.Env] = None,
                 tag: str = "default",
                 seed: int = 1,
                 device: str = "cpu",
                 num_steps: int = 80,
                 num_actors: int = 45,
                 num_learners: int = 4,
                 num_storages: int = 60,
                 batch_size: int = 4
                 ) -> None:
        super().__init__(env=env,
                         eval_env=eval_env,
                         tag=tag,
                         seed=seed,
                         device=device,
                         pretraining=False
                         )
        self.num_actors = num_actors
        self.num_learners = num_learners
        self.num_steps = num_steps
        self.num_storages = num_storages
        self.batch_size = batch_size
        # get separate environments
        self.env = self.env.envs

    def train(self, num_train_steps: int = 30000000, init_model_path: Optional[str] = None) -> None:
        """Training function.
        
        Args:
            num_train_steps (int): Number of training steps.
            init_model_path (Optional[str]): Path of Iinitial model parameters.

        Returns:
            None.
        """
        # freeze the structure of the agent
        self.freeze()
        # final check
        self.check()
        # load initial model parameters 
        if init_model_path is not None:
            self.load(init_model_path)
        
        def lr_lambda(epoch):
            return (
                1.0 - min(epoch * self.num_steps * self.num_learners, num_train_steps) / num_train_steps
            )
        self.lr_scheduler = th.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

        """Training function"""
        global_step = 0
        global_episode = 0
        metrics = dict()
        episode_rewards = deque(maxlen=10)
        episode_steps = deque(maxlen=10)

        def sample_and_update(i, lock=threading.Lock()):  # noqa: B008
            """Thread target for the learning process."""
            nonlocal global_step, global_episode, metrics
            while global_step < num_train_steps:
                batch, actor_states = self.storage.sample(
                    device=self.device,
                    batch_size=self.batch_size,
                    free_queue=free_queue,
                    full_queue=full_queue,
                    storages=self.storage.storages,
                    init_actor_state_storages=init_actor_state_storages,
                )
                metrics = self.update(
                    batch=batch,
                    init_actor_states=actor_states,
                    lr_scheduler=self.lr_scheduler,
                )
                with lock:
                    global_step += self.num_steps * self.batch_size
                    global_episode += self.batch_size

        # TODO: Add initial RNN state.
        init_actor_state_storages = []
        for _ in range(self.num_storages):
            state = self.actor.init_state(batch_size=1)
            for t in state:
                t.share_memory_()
            init_actor_state_storages.append(state)

        actor_pool = []
        ctx = mp.get_context("fork")
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()

        for actor_idx in range(self.num_actors):
            actor = ctx.Process(
                target=self.act,
                kwargs={
                    "env": Environment(self.env[actor_idx]),
                    "actor_idx": actor_idx,
                    "free_queue": free_queue,
                    "full_queue": full_queue,
                    "init_actor_state_storages": init_actor_state_storages,
                },
            )
            actor.start()
            actor_pool.append(actor)
        self.logger.info(f"{self.num_actors} actors started!")

        for m in range(self.num_storages):
            free_queue.put(m)

        threads = []
        for i in range(self.num_learners):
            thread = threading.Thread(target=sample_and_update, name="sample-and-update-%d" % i, args=(i,))
            thread.start()
            threads.append(thread)
        self.logger.info(f"{self.num_learners} learners started!")

        try:
            log_times = 0
            while global_step < num_train_steps:
                start_step = global_step
                time.sleep(5)

                if metrics.get("episode_returns"):
                    episode_rewards.extend(metrics["episode_returns"])
                    episode_steps.extend(metrics["episode_steps"])

                if len(episode_rewards) > 0:
                    episode_time, total_time = self.timer.reset()
                    train_metrics = {
                        "step": global_step,
                        "episode": global_episode,
                        "episode_length": np.mean(episode_steps),
                        "episode_reward": np.mean(episode_rewards),
                        "fps": (global_step - start_step) / episode_time,
                        "total_time": total_time,
                    }
                    self.logger.train(msg=train_metrics)
                    log_times += 1

                # if log_times % 50 == 0:
                #     episode_time, total_time = self.timer.reset()
                #     test_metrics = self.test()
                #     test_metrics.update({
                #         "step": global_step,
                #         "episode": global_episode,
                #         "total_time": total_time,
                #         })
                #     self.logger.test(msg=test_metrics)

        except KeyboardInterrupt:
            # TODO: join actors then quit.
            return
        else:
            for thread in threads:
                thread.join()
            self.logger.info("Training Accomplished!")
            # save model
            self.save()
        finally:
            for _ in range(self.num_actors):
                free_queue.put(None)
            for actor in actor_pool:
                actor.join(timeout=1)

    def eval(self) -> Dict[str, float]:
        """Evaluation function."""
        env = Environment(self.eval_env.envs[0])
        seed = self.num_actors * int.from_bytes(os.urandom(4), byteorder="little")
        env_output = env.reset(seed)

        episode_rewards = list()
        episode_steps = list()
        while len(episode_rewards) < self.num_eval_episodes:
            with th.no_grad():
                actor_output, _ = self.actor.get_action(env_output, training=False)
            env_output = env.step(actor_output["action"])
            if env_output["terminated"].item() or env_output["truncated"].item():
                episode_rewards.append(env_output["episode_return"].item())
                episode_steps.append(env_output["episode_step"].item())

        return {
            "episode_length": np.mean(episode_steps),
            "episode_reward": np.mean(episode_rewards),
        }