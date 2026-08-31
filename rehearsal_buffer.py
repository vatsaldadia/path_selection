import torch
import random

class RehearsalBuffer:
    def __init__(self, capacity=5120):
        self.capacity = capacity
        self.observed_count = 0  # Total samples seen so far
        self.storage = {}

    def update(self, obs):

        if not self.storage:
            for k, v in obs.items():
                v_tensor = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                self.storage[k] = torch.zeros((self.capacity, *v_tensor.shape),
                                              dtype=v_tensor.dtype, device='cpu')

        add = True
        if self.observed_count < self.capacity:
            idx = self.observed_count
        else:
            if random.random() < 0.5:
                idx = random.randint(0, self.capacity - 1)
            else:
                add = False
        if add:
            for k, v in obs.items():
                self.storage[k][idx] = torch.tensor(v)

            self.observed_count += 1

    def sample(self, sample_size=1024):
        actual_size = min(self.observed_count, self.capacity)
        indices = torch.randperm(actual_size)[:sample_size]
        return {k: v[indices] for k, v in self.storage.items()}