import torch
from stable_baselines3.common.callbacks import BaseCallback

import os
import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import EvalCallback
import tempfile  # To save normalisation stats temporarily
from stable_baselines3.common.callbacks import BaseCallback
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import TensorBoardOutputFormat

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from torch.utils.tensorboard import SummaryWriter


class DebugLoggingCallback(BaseCallback):
    # def __init__(self, log_name: str):
    #     super().__init__()
    #     self.log_name = log_name

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        for key in info.keys():
            self.logger.record(f"{key}", info[key])

        replay_buffer = self.model.replay_buffer
        if replay_buffer.full or replay_buffer.pos > self.model.batch_size:
            replay_data = replay_buffer.sample(self.model.batch_size, env=self.model._vec_normalize_env)

            with torch.no_grad():
                q_values = self.model.q_net(replay_data.observations)
                mean_q_value = torch.mean(torch.max(q_values, dim=1).values).item()

            self.logger.record("rollout/mean_q_value", mean_q_value)

        if self.n_calls % 5000 == 0:
            tb_formatter = None
            for output_format in self.logger.output_formats:
                if isinstance(output_format, TensorBoardOutputFormat):
                    tb_formatter = output_format
                    break

            if tb_formatter is not None:
                # Access the underlying SummaryWriter instance
                writer = tb_formatter.writer
                q_net = self.model.q_net
                for name, param in q_net.named_parameters():
                    if param is not None:
                        writer.add_histogram(f"q_net_weights/{name}", param.data, self.num_timesteps)
                        if param.grad is not None:
                            writer.add_histogram(f"q_net_grads/{name}", param.grad, self.num_timesteps)
                        param_norm = param.data.norm(2).item()
                        self.logger.record(f"q_net_norms/{name}", param_norm)

        return True

class MetricsEvalCallback(EvalCallback):
    """
    This is the definitive callback for robust, windowed evaluation.
    - It correctly subclasses MaskableEvalCallback as required.
    - It re-creates the eval_env from scratch for each evaluation to ensure an unbiased test.
    - It uses an inner callback to correctly gather per-episode metrics.
    - It saves the best model based on feasibility_rate and supports early stopping.
    """

    def __init__(self, eval_env, eval_env_fn, **kwargs):
        # We need the function to create the env, and we pass the parent's kwargs.
        # We initialize the parent with eval_env=None, as we create it dynamically.
        super(MetricsEvalCallback, self).__init__(eval_env=eval_env, **kwargs)
        self.eval_env_fn = eval_env_fn

        # --- Our custom state variables ---
        self._info_buffer = []
        self.n_evaluations = 0
        self.best_mean_feasibility = -1.0
        self.patience = 10
        self.patience_counter = 0

        self.loaded_traffic = None

    def _log_info_callback(self, locals: dict, globals: dict) -> None:
        """
        This inner callback is called at the end of each episode.
        The `info` dict it receives is the final one we need.
        """
        is_done = locals.get("dones")[0]

        if is_done:
            # The episode has just finished. Grab the info dict.
            info = locals.get("info")
            if info is not None:
                # The info object is a list in a VecEnv, get the first element.
                # Then append a copy to our buffer.
                self._info_buffer.append(info.copy())

    def _on_step(self) -> bool:
        """
        Override this method to inject env-recreation and custom metric logic.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # --- 1. Re-create the evaluation environment for a fresh, unbiased test ---
            # if self.verbose > 0:
            # print(f"--- Running evaluation #{self.n_evaluations + 1} @ Timestep {self.num_timesteps} ---")

            # Use a new seed for every evaluation run
            current_seed = self.n_evaluations + 100
            eval_env = make_vec_env(self.eval_env_fn(loaded_traffic=self.loaded_traffic, phase="evaluation"), n_envs=1, seed=current_seed)
            eval_env = VecMonitor(eval_env)

            # Apply the learned normalization stats from the training env
            if isinstance(self.training_env, VecNormalize):
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
                    stats_path = tmp_file.name
                    self.training_env.save(stats_path)
                    eval_env = VecNormalize.load(stats_path, eval_env)
                os.remove(stats_path)
                eval_env.training = False
                eval_env.norm_reward = False

            # --- 2. Run Evaluation on the new environment ---
            self._info_buffer = []  # Reset our buffer before the run

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,                     # type: ignore[arg-type]
                eval_env,  # Use the new, temporary environment
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
                callback=self._log_info_callback,  # Pass our inner callback
            )
            eval_env.close()  # Close the temporary environment

            # --- 3. Calculate and Log All Metrics for the Window ---
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            mean_ep_length = np.mean(episode_lengths) if episode_lengths else 0.0

            # Count successes by checking the info dict from each episode
            metrics_last_info = self._info_buffer[-1]
            successful_episodes = metrics_last_info["granted"]
            feasibility_rate = metrics_last_info["GoS"]
            target_dead_ends = metrics_last_info["target_dead_end"]
            latency_dead_ends = metrics_last_info["latency_dead_end"]
            bandwidth_dead_ends = metrics_last_info["bandwidth_dead_end"]
            processing_time_violations = metrics_last_info["processing_time_violation"]

            # print(f"Feasibility Rate (this window): {feasibility_rate:.3f}")

            # Log everything to TensorBoard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/feasibility_rate", feasibility_rate)
            self.logger.record("eval/successful_episodes", successful_episodes)
            self.logger.record("eval/target_dead_ends", target_dead_ends)
            self.logger.record("eval/latency_dead_ends", latency_dead_ends)
            self.logger.record("eval/bandwidth_dead_ends", bandwidth_dead_ends)
            self.logger.record("eval/processing_time_violations", processing_time_violations)
            self.logger.dump(step=self.num_timesteps)

            # --- 4. Save Best Model & Handle Early Stopping ---
            if feasibility_rate > self.best_mean_feasibility:
                # print(f"New best model found! Feasibility rate: {feasibility_rate:.3f}")
                self.best_mean_feasibility = feasibility_rate
                if self.best_model_save_path:
                    os.makedirs(self.best_model_save_path, exist_ok=True)
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    if isinstance(self.training_env, VecNormalize):
                        self.training_env.save(os.path.join(self.best_model_save_path, "vec_normalize.pkl"))
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                # print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")

            self.n_evaluations += 1

            if self.patience_counter >= self.patience:
                print(f"Stopping early with best feasibility rate: {self.best_mean_feasibility:.3f}")
                return False

        return True