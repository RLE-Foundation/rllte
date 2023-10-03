# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


import os
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np
import torch as th
from torch import multiprocessing as mp

from rllte.common.prototype.base_agent import BaseAgent
from rllte.common.type_alias import DistributedPolicyType, DistributedStorageType, VecEnv
from rllte.env.utils import DistributedWrapper


class DistributedAgent(BaseAgent):  # type: ignore
    """Trainer for distributed algorithms.

    Args:
        env (VecEnv): Vectorized environments for training.
        eval_env (VecEnv): Vectorized environments for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_steps (int): The sample length of per rollout.
        num_actors (int): Number of actors.
        num_learners (int): Number of learners.
        num_storages (int): Number of storages.
        **kwargs: Arbitrary arguments such as `batch_size` and `hidden_dim`.

    Returns:
        Distributed agent instance.
    """

    def __init__(
        self,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        num_steps: int = 80,
        num_actors: int = 45,
        num_learners: int = 4,
        num_storages: int = 60,
        **kwargs,
    ) -> None:
        super().__init__(env=env, eval_env=eval_env, tag=tag, seed=seed, device=device, pretraining=False)

        self.num_actors = num_actors
        self.num_learners = num_learners
        self.num_steps = num_steps
        self.num_storages = num_storages

        # get separate environments
        try:
            self.env = self.env.envs  # type: ignore
        except AttributeError:
            raise AttributeError("Asynchronous execution is unavailable for distributed training!")  # noqa: B904

        # create process and thread pool
        self.ctx = mp.get_context("fork")
        self.free_queue = self.ctx.SimpleQueue()
        self.full_queue = self.ctx.SimpleQueue()
        self.actor_pool: List = list()
        self.learner_threads: List[threading.Thread] = list()

        # linear learning rate scheduler
        def lr_lambda(num_train_steps: int) -> Callable:
            """Function for learning rate scheduler.

            Args:
                num_train_steps (int): The number of training steps.

            Returns:
                Learning rate.
            """
            return lambda epoch: 1.0 - min(epoch * self.num_steps * self.num_learners, num_train_steps) / num_train_steps

        self.lr_lambda = lr_lambda

        # attr annotations
        self.policy: DistributedPolicyType
        self.storage: DistributedStorageType

    def run(self, env: DistributedWrapper, actor_idx: int) -> None:
        """Sample function of each actor. Implemented by individual algorithms.

        Args:
            env (DistributedWrapper): A Gym-like environment wrapped by `DistributedWrapper`.
            actor_idx (int): The index of actor.

        Returns:
            None.
        """
        try:
            # reset environment
            seed = actor_idx * int.from_bytes(os.urandom(4), byteorder="little")
            env_output = env.reset(seed)
            # get initial actor output
            actor_output = self.policy.actor(env_output, training=True)

            while True:
                idx = self.free_queue.get()
                if idx is None:
                    break

                # write old rollout end.
                self.storage.add(idx, 0, actor_output, env_output)
                # do new rollout.
                for t in range(self.num_steps):
                    with th.no_grad():
                        actor_output = self.policy.actor(env_output, training=True)
                    env_output = env.step(actor_output["actions"])

                    self.storage.add(idx, t + 1, actor_output, env_output)

                self.full_queue.put(idx)

        # return silently.
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.logger.error(f"Exception in worker process {actor_idx}!")
            traceback.print_exc()
            raise e

    def update(self, *args, **kwargs) -> None:
        """Update the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def train(  # noqa: C901
        self,
        num_train_steps: int,
        init_model_path: Optional[str] = None,
        log_interval: int = 1,
        eval_interval: int = 5000,
        save_interval: int = 5000,
        num_eval_episodes: int = 10,
        th_compile: bool = False,
    ) -> None:
        """Training function.

        Args:
            num_train_steps (int): The number of training steps.
            init_model_path (Optional[str]): The path of the initial model.
            log_interval (int): The interval of logging.
            eval_interval (int): The interval of evaluation.
            save_interval (int): The interval of saving model.
            num_eval_episodes (int): The number of evaluation episodes.
            th_compile (bool): Whether to use `th.compile` or not.

        Returns:
            None.
        """
        # freeze the agent and get ready for training
        self.freeze(init_model_path=init_model_path, th_compile=th_compile)

        # set learning rate scheduler
        self.lr_scheduler = th.optim.lr_scheduler.LambdaLR(self.policy.optimizers["opt"], self.lr_lambda(num_train_steps))

        # training tracker
        global_step = 0
        global_episode = 0
        metrics = dict()
        episode_rewards: Deque = deque(maxlen=10)
        episode_steps: Deque = deque(maxlen=10)

        def sample_and_update(lock=threading.Lock()):  # B008
            """Thread target for the learning process."""
            nonlocal global_step, global_episode, metrics
            while global_step < num_train_steps:
                # sample batch
                batch = self.storage.sample(free_queue=self.free_queue, full_queue=self.full_queue)
                # update agent
                metrics = self.update(batch)
                with lock:
                    global_step += self.num_steps * self.storage.batch_size
                    global_episode += self.storage.batch_size

        # start actor processes
        for actor_idx in range(self.num_actors):
            actor = self.ctx.Process(  # type: ignore
                target=self.run,
                kwargs={"env": DistributedWrapper(self.env[actor_idx]), "actor_idx": actor_idx},  # type: ignore
            )
            actor.start()
            self.actor_pool.append(actor)  # type: ignore
        self.logger.info(f"{self.num_actors} actors started!")

        # serialize the data
        for _ in range(self.num_storages):
            self.free_queue.put(_)

        # start learner threads
        for i in range(self.num_learners):
            thread = threading.Thread(target=sample_and_update, name=f"sample-and-update-{i}")
            thread.start()
            self.learner_threads.append(thread)
        self.logger.info(f"{self.num_learners} learners started!")

        try:
            log_times = 0
            log_available = False
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
                        "episode_length": np.mean(list(episode_steps)),
                        "episode_reward": np.mean(list(episode_rewards)),
                        "fps": (global_step - start_step) / episode_time,
                        "total_time": total_time,
                    }
                    log_available = True
                    log_times += 1

                if (log_times % log_interval == 0) and log_available:
                    self.logger.train(msg=train_metrics)

                # if log_times % eval_interval == 0:
                #     episode_time, total_time = self.timer.reset()
                #     eval_metrics = self.eval(num_eval_episodes)
                #     eval_metrics.update({
                #         "step": global_step,
                #         "episode": global_episode,
                #         "total_time": total_time,
                #         })
                #     self.logger.eval(msg=eval_metrics)

                # save model
                if global_episode % save_interval == 0:
                    self.save()
            # final save
            self.save()

        except KeyboardInterrupt:
            # TODO: join actors then quit.
            return
        else:
            for thread in self.learner_threads:
                thread.join()
            self.logger.info("Training Accomplished!")
            self.logger.info(f"Model saved at: {self.work_dir / 'model'}")
        finally:
            for _ in range(self.num_actors):
                self.free_queue.put(None)
            for actor in self.actor_pool:
                actor.join(timeout=1)

    def eval(self, num_eval_episodes: int) -> Dict[str, Any]:
        """Evaluation function.

        Args:
            num_eval_episodes (int): The number of evaluation episodes.

        Returns:
            The evaluation results.
        """
        assert self.eval_env is not None, "Please set `eval_env` for evaluation!"
        env = DistributedWrapper(self.eval_env.envs[0])
        seed = self.num_actors * int.from_bytes(os.urandom(4), byteorder="little")
        env_output = env.reset(seed)

        episode_rewards: List[float] = []
        episode_steps = []
        while len(episode_rewards) < num_eval_episodes:
            with th.no_grad():
                actor_output = self.policy.actor(env_output, training=False)
            env_output = env.step(actor_output["actions"])
            if env_output["terminateds"].item() or env_output["truncateds"].item():
                episode_rewards.append(env_output["episode_returns"].item())
                episode_steps.append(env_output["episode_steps"].item())

        return {"episode_lengths": np.mean(episode_steps), "episode_rewards": np.mean(episode_rewards)}
