import torch as th
import torch.nn as nn
import numpy as np
import os
import json
import copy

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import QNetwork
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from callbacks import DebugLoggingCallback
from drift_detector import DriftDetector
from env.path_selection_env import PathSelectionEnv
from generator import Generator
from maskable_dqn import MaskableDQN
from maskable_features_extractor import MaskableFeaturesExtractor
from maskable_multi_input import MaskableMultiInputPolicy
from maskable_qnetwork import MaskableQNetwork
from rehearsal_buffer import RehearsalBuffer

from typing import Any, Dict, Optional, Tuple, Type, Union, Callable

SEED = 42

class Task:

    def __init__(self, first_hour: int, env: Union[gym.Env, VecEnv]):
        self.first_hour = first_hour
        self.env = env
        self.model = MaskableDQN(
            policy=MaskableMultiInputPolicy,
            env=env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=128,
            gamma=0.995,
            tau=1,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10000,
            exploration_fraction=0.2,
            verbose=0,
            tensorboard_log="./tensorboard_logs/",
            policy_kwargs=dict(
                features_extractor_class=MaskableFeaturesExtractor,
                net_arch=[64, 64],
                activation_fn=nn.ReLU,
            ),
            device="auto",
            seed=SEED,
        )

class ContinualDQN(nn.Module):

    def __init__(
            self,
            generator: Generator,
            detector: DriftDetector,
            job_id: str,
            mode: str = "continual",
            timesteps: int = 500_000,
            epochs: int = 50_000,
            ewc_lambda: float = 0.4,
            dynamic_end_times: bool = False
    ):
        super().__init__()

        if mode == "continual":
            self.continual_flag = True
        elif mode == "dqn":
            self.continual_flag = False
        else:
            raise ValueError("mode must be either 'continual' or 'dqn'")

        self.generator = generator
        self.detector = detector
        self.timesteps = timesteps
        self.epochs = epochs
        self.ewc_lambda = ewc_lambda

        self.q_net: Optional[nn.Sequential] = None
        self.features_extractor: Optional[BaseFeaturesExtractor] = None

        self.past_tasks: list[Task] = []
        self.current_task: Task | None = None

        device = "cuda:0" if th.cuda.is_available() else "cpu"
        self.to(device)

        self.dynamic_end_times = dynamic_end_times
        self.job_id = job_id
        if self.continual_flag:
            self.writer = SummaryWriter(f'./tensorboard_logs/{job_id}')

        # self.optimizer = th.optim.Adam(self.parameters(), lr=1e-4)
        self.optimizer = None

    def change_task(self) -> None:
        print("Changing task.....")
        if self.current_task is not None:
            self.past_tasks.append(self.current_task)

        if not self.generator.can_do_new_task():
            print("No new task available from generator, stopping training.")
            self.current_task.model.policy.q_net.load_state_dict(self.state_dict())
            save_path = f"./models/{self.job_id}/"
            os.makedirs(save_path, exist_ok=True)
            self.current_task.model.save(f"{save_path}/final_model.zip")
            return

        env = self.create_env(self.generator.env_last_hour + 1, phase="training")           # NOTE: 1 env handles 1 traffic csv file
        self.current_task = Task(self.generator.env_last_hour + 1, env)
        if not self.continual_flag and self.job_id.find(",") != -1:
            prev_job = ",".join(self.job_id.split(",")[:-1]) + "#"
            if os.path.exists(f"./models/{prev_job}"):
                self.current_task.model = MaskableDQN.load(f"./models/{prev_job}/final_model.zip", env=env)
            else:
                raise ValueError(f"Failed {self.job_id}. Train {prev_job} first when continual_flag is {self.continual_flag}.")
        self.progress()

    def make_env(self, env_first_hour, phase):
        if phase not in ["training", "evaluation", "testing"]:
            raise ValueError("phase must be one of ['training', 'evaluation', 'testing']")
        def _init():
            env = PathSelectionEnv(env_first_hour=env_first_hour, generator=self.generator, detector=self.detector,
                                   phase=phase, dynamic_end_times=self.dynamic_end_times)
            return env

        return _init

    def create_env(self, env_first_hour: int , phase: str, seed: int = SEED) -> VecEnv:
        env = make_vec_env(self.make_env(env_first_hour=env_first_hour, phase=phase), n_envs=1,
                                    seed=seed)
        return env

    def forward(self, obs: th.Tensor) -> th.Tensor:
        assert self.features_extractor is not None
        extracted_features = self.features_extractor(obs)

        assert self.q_net is not None
        q_values = self.q_net(extracted_features)

        return q_values

    def progress(self):

        # save_path = f"./models/{self.job_id}/"
        result_path = f"./results/{self.job_id}/"
        # os.makedirs(save_path, exist_ok=True)
        os.makedirs(result_path, exist_ok=True)

        print(f"Starting training on task with first hour {self.current_task.first_hour}")

        timesteps = self.timesteps

        debug_callback = DebugLoggingCallback()
        assert self.current_task is not None
        self.current_task.model.learn(total_timesteps=timesteps,
                                      tb_log_name=f"{self.job_id}",
                                      callback=debug_callback,
                                      progress_bar=True)

        # self.current_task.model.save(save_path + "best_model.zip")

        metrics = self.current_task.env.env_method("return_metrics")[0]  # get metrics from the first env
        with open(f"{result_path}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"{metrics}")

        self.compress()

    def _compute_fisher(self):
        fisher = {}
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}

        for n, p in params.items():
            fisher[n] = th.zeros_like(p.data)

        self.eval()

        for past_task in self.past_tasks:               # use data from previous tasks to compute importance of parameters
            buffer = past_task.env.env_method("get_rehearsal_buffer")[0]
            obs = buffer.sample(1024)
            device = "cuda:0" if th.cuda.is_available() else "cpu"
            obs = {k: v.to(device) for k, v in obs.items()}
            outputs = self(obs)
            loss = th.mean(outputs ** 2)             # calculate "importance" via gradients
            self.zero_grad()
            loss.backward()

            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(self.past_tasks)      # calculate average importance across all previous tasks

        fisher = {n: p.detach() for n, p in fisher.items()}
        snapshot_weights = {n: p.clone().detach() for n, p in self.named_parameters()}

        return fisher, snapshot_weights

    def compress(self):

        assert self.current_task is not None
        if not self.past_tasks:
            print("No past tasks available, copying weights...")
            self.q_net = copy.deepcopy(self.current_task.model.policy.q_net.q_net)
            self.features_extractor = copy.deepcopy(self.current_task.model.policy.q_net.features_extractor)
            self.optimizer = th.optim.Adam(self.parameters(), lr=1e-4)
        else:
            print("Compressing continual model")
            fisher, snapshot_weights = self._compute_fisher()
            device = "cuda:0" if th.cuda.is_available() else "cpu"

            for epoch in range(self.epochs):
                buffer = self.current_task.env.env_method("get_rehearsal_buffer")[0]
                obs = buffer.sample(128)
                obs = {k: v.to(device) for k, v in obs.items()}
                with th.no_grad():
                    target_logits = self.current_task.model.policy.q_net(obs)               # TODO: ensure sampled obs not added back to rehearsal buffer
                current_logits = self(obs)
                distil_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(current_logits, dim=1),
                    nn.functional.softmax(target_logits, dim=1),
                    reduction='batchmean'
                )

                ewc_loss = 0
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        fisher_n = fisher.get(n, th.zeros_like(p))
                        snapshot_n = snapshot_weights.get(n, th.zeros_like(p))
                        ewc_loss += (fisher_n * (p - snapshot_n) ** 2).sum()

                loss = distil_loss + self.ewc_lambda * ewc_loss

                self.writer.add_scalar("loss_ewc/train", loss.item(), epoch + (self.epochs * len(self.past_tasks)))

                if (epoch + 1) % 10_000 == 0:
                    print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("Compression complete, moving to next task")

        self.change_task()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.change_task()

    def test(self, episodes=24):
        self.eval()

        model_path = f"./models/{self.job_id}/"
        env = self.create_env(0, phase="testing", seed=SEED + 1)
        model = MaskableDQN.load(model_path + "final_model", env=env)
        del model.policy.q_net_target
        test_mean_reward, test_episode_lengths = evaluate_policy(
            model,  # type: ignore[arg-type]
            env,
            n_eval_episodes=episodes,
            deterministic=True,
            warn=False,
        )

        result_path = f"./results/{self.job_id}/"
        metrics = env.env_method("return_metrics")[0]  # get metrics from the first env
        print(metrics)
        with open(f"{result_path}/testing_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        test_results = env.env_method("return_hourly_results")[0]
        with open(f"{result_path}/testing_hourly_results.json", "w") as f:
            json.dump(test_results, f, indent=4)

        cpu_results = env.env_method("return_cpu_metrics")[0]
        with open(f"{result_path}/testing_cpu_results.json", "w") as f:
            json.dump(cpu_results, f, indent=4)