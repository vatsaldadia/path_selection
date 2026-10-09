import json

import torch as th
import gymnasium as gym
import numpy as np
from torch.profiler import profile, ProfilerActivity
from typing import Dict, Any


class FLOPSTracker:
    def __init__(self):
        self.observation_space = None
        self.unit_costs = {
            "rl_train_step": 0,
            "compression_epoch": 0,
            "fisher_step": 0,
        }
        self.counters = {
            "env_steps": 0,
            "rl_train_steps": 0,
            "compression_epochs": 0,
            "fisher_runs": 0,
        }
        self._benchmarked_rl = False
        self._benchmark_fisher = False
        self._benchmarked_compression = False

    def set_observation_space(self, observation_space: gym.spaces.Space):
        self.observation_space = observation_space

    def generate_dummy_obs(self, device: str = "cpu", batch_size: int = 1) -> Dict[str, th.Tensor]:
        """Generates batched PyTorch tensors matching your Dict observation space layout."""
        dummy_obs = {}
        for key, space in self.observation_space.spaces.items():
            if isinstance(space, gym.spaces.Discrete):
                dummy_obs[key] = th.zeros((batch_size,), dtype=th.long, device=device)
            elif isinstance(space, gym.spaces.MultiBinary):
                shape = (batch_size, space.n) if hasattr(space, 'n') else (batch_size, space.shape[0])
                dummy_obs[key] = th.zeros(shape, dtype=th.float32, device=device)
            elif isinstance(space, gym.spaces.Box):
                shape = (batch_size,) + space.shape
                dummy_obs[key] = th.zeros(shape, dtype=th.float32, device=device)
        return dummy_obs

    def _compute_linear_flops(self, module: th.nn.Module, batch_size: int, is_training: bool = True) -> int:
        """
        Analytically calculates FLOPs for all linear layers in a module.
        Formula per linear layer:
          - Forward pass = 2 * batch_size * in_features * out_features
          - Backward pass = ~2x forward pass (total training step = ~3x forward)
        """
        total_flops = multiplier = 0
        try:
            for module in module.modules():
                if isinstance(module, th.nn.Linear):
                    fwd_flops = 2 * batch_size * module.in_features * module.out_features
                    # Training does forward + backward (weights grad + input grad), roughly 3x forward
                    # Inference (evaluation/forward pass) is just 1x forward
                    multiplier = 3 if is_training else 1
                    total_flops += fwd_flops * multiplier
        except Exception as e:
            print(f"Error computing analytical FLOPs: {e}")
            # Safe fallback default based on your [64, 64] net_arch
            total_flops = batch_size * 50000 * (3 if is_training else 1)

        return total_flops

    def benchmark_rl(self, model: Any):
        """Calculates analytical unit costs for CPU execution."""
        if self._benchmarked_rl:
            return

        print("Calculating analytical FLOP costs for CPU execution...")
        sb3_policy = model.current_task.model.policy
        batch_size = model.current_task.model.batch_size

        # 1. Standard RL Training Step (Forward + Backward pass for a batch)
        self.unit_costs["rl_train_step"] = self._compute_linear_flops(sb3_policy, batch_size=batch_size, is_training=True)

        # print(f"CPU Analytical Unit Costs (Inference & RL): {self.unit_costs}")
        self._benchmarked_rl = True

    def benchmark_fisher(self, continual_model: Any):
        """Benchmarks a single Fisher Information Matrix calculation pass (batch size 1024)."""
        if self._benchmark_fisher or continual_model.q_net is None:
            return

        if self._benchmarked_compression:
            return
        batch_size = 1024
        self.unit_costs["fisher_step"] = self._compute_linear_flops(continual_model, batch_size=batch_size,
                                                                    is_training=True)
        self._benchmark_fisher = True

    def benchmark_compression(self, continual_model: Any):
        """Benchmarks the ContinualDQN compression epoch (distillation + EWC) once q_net is initialised."""
        if self._benchmarked_compression or continual_model.q_net is None:
            return

        print("Benchmarking ContinualDQN Compression Epoch...")
        # continual_model.train()

        # Compression epoch does: Target forward (no grad) + Current forward + Distillation backward + EWC backward
        # Roughly double the work of a standard training step for batch size 128
        batch_size = 128
        base_step = self._compute_linear_flops(continual_model, batch_size=batch_size, is_training=True)
        self.unit_costs["compression_epoch"] = int(base_step * 1.5)  # EWC + KL divergence overhead
        # print(f"CPU Analytical Unit Cost (Compression Epoch): {self.unit_costs['compression_epoch']}")
        self._benchmarked_compression = True

    def increment(self, counter_name: str, amount: int = 1):
        if counter_name in self.counters:
            self.counters[counter_name] += amount

    def write_summary(self, path: str):
        total_rl_flops = self.counters["rl_train_steps"] * self.unit_costs["rl_train_step"]
        total_fisher_flops = self.counters["fisher_runs"] * self.unit_costs["fisher_step"]
        total_compression_flops = self.counters["compression_epochs"] * self.unit_costs["compression_epoch"]

        grand_total = total_rl_flops + total_fisher_flops + total_compression_flops

        return_dict = {
            "unit_benchmarks": self.unit_costs,
            "counters": self.counters,
            "breakdown_flops": {
                "rl_training_total": total_rl_flops,
                "fisher_total": total_fisher_flops,
                "compression_total": total_compression_flops
            },
            "cumulative_total_flops": grand_total
        }

        with open(path, "w") as f:
            json.dump(return_dict, f)