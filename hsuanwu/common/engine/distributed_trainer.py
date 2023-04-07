import os
os.environ["OMP_NUM_THREADS"] = "1" 
import threading
import time
import traceback

import hydra
import torch
from torch import multiprocessing as mp

from hsuanwu.common.engine import BasePolicyTrainer
from hsuanwu.common.logger import *
from hsuanwu.common.typing import DictConfig, Env, NNModule

import timeit, pprint

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
        # xploit part
        self._learner = hydra.utils.instantiate(self._cfgs.learner)
        # TODO: build storage
        self._shared_storages = hydra.utils.instantiate(self._cfgs.storage)

        self._train_env = self._train_env.envs

    @staticmethod
    def act(
        cfgs: DictConfig,
        logger: Logger,
        actor_idx: int,
        env: Env,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        actor_model: NNModule,
        storages: List[Storage],
        init_actor_states: List[Tensor],
        ):
        try:
            logger.log(INFO, f"Actor {actor_idx} started.")

            env_output = env.initial()
            actor_state = actor_model.initial_state(batch_size=1)
            actor_output, _ = actor_model(env_output, actor_state)
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
                        actor_output, actor_state = actor_model(env_output, actor_state)
                    env_output = env.step(actor_output["action"])

                    for key in env_output:
                        storages[key][idx][t + 1, ...] = env_output[key]
                    for key in actor_output:
                        storages[key][idx][t + 1, ...] = actor_output[key]

                full_queue.put(idx)

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            logger.log(INFO, f"Exception in worker process {actor_idx}!")
            traceback.print_exc()
            raise e

    def train(self):
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
                    lr_scheduler=self._learner.lr_scheduler
                )
                with lock:
                    global_step += self._cfgs.num_steps * self._cfgs.storage.batch_size
                    # self._global_step += self._cfgs.num_steps * self._cfgs.storage.batch_size

        # TODO: Add initial RNN state.
        init_actor_states = []
        for _ in range(self._cfgs.storage.num_storages):
            state = self._learner.actor.initial_state(batch_size=1)
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
                self._logger.log(INFO, f"Steps {global_step} @ {sps} SPS. Loss {total_loss}. {mean_return}Stats:\n{pprint.pformat(metrics)}")
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
        pass

    def save(self) -> None:
        pass