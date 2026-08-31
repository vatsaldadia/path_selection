import heapq
from typing import Optional

import gymnasium as gym
import random
import torch as th

from drift_detector import DriftDetector
from env.engine import Engine
from env.obs_builder import ObservationBuilder
from env.traffic_flow import TrafficFlow
from env.metrics_collector import MetricsCollector
from generator import Generator
from rehearsal_buffer import RehearsalBuffer


class PathSelectionEnv(gym.Env):

    def __init__(self,
                 env_first_hour,
                 generator: Generator,
                 detector: DriftDetector,
                 phase: str,
                 dynamic_end_times=False):
        super().__init__()

        self.env_first_hour = env_first_hour
        self.generator = generator
        self.detector = detector
        self.phase = phase
        self.rehearsal_buffer = RehearsalBuffer() if phase == "training" else None
        self.dynamic_end_times = dynamic_end_times

        self.current_step = 0
        self.active_flows = []
        start_hour = -1
        self.iteration_hour = start_hour
        self.traffic = []
        self.env_last_hour = -1

        self.engine = Engine(network_data=generator.get_network_data())
        self.metrics_collector = MetricsCollector()
        self.obs_builder = ObservationBuilder(engine=self.engine)
        self.observation_space = self.obs_builder.observation_space
        self.action_space = gym.spaces.Discrete(self.engine.action_space_dim)

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed, options=options)

        if self.phase != "evaluation":
            self.metrics_collector.update_cpu_metrics(self.iteration_hour, self.engine.node_data)
        if not self.dynamic_end_times:
            self.engine.reset_network_resources()
        self.iteration_hour += 1                                       # to keep track of total hours passed
        if self.phase == "training":                        # drift detection only during training
            if self.env_last_hour == -1:
                if self.detector.is_drift(self.iteration_hour):
                    # raise ValueError("Drift detected. This should not happen.")
                    self.env_last_hour = self.env_first_hour + self.iteration_hour - 1
                    self.generator.update_env_hour(self.env_last_hour)
                    traffic_hour = self.env_first_hour
                else:
                    traffic_hour = self.env_first_hour + self.iteration_hour
            else:
                traffic_hour = self.iteration_hour % (
                            self.env_last_hour - self.env_first_hour + 1) + self.env_first_hour
            # print(f"{self.env_first_hour=}, {self.env_last_hour=}, {self.iteration_hour=}, {traffic_hour=}")
            self.traffic = self.generator.get_hour_traffic(traffic_hour)
        else:
            self.traffic = self.generator.get_hour_traffic(self.env_first_hour + self.iteration_hour)
            # print(self.iteration_hour, len(self.traffic))

        return self.obs_builder.create_obs_dict(flow=self.traffic.pop(0) if self.traffic else None), {}

    def step(self, action):
        self.current_step += 1

        current_flow = self.obs_builder.current_flow
        assert current_flow is not None
        self.metrics_collector.update_hourly_metrics(self.iteration_hour, current_flow, "encountered")

        self.rehearsal_buffer.update(self.obs_builder.obs) if self.rehearsal_buffer else None

        mask = self.obs_builder.obs["action_mask"]
        result = self.engine.make_step(current_flow, action, mask)

        self.metrics_collector.update_hourly_metrics(self.iteration_hour, current_flow, result.description)

        if current_flow.is_granted():
            current_flow.power_consumed = self.engine.get_power(current_flow)
            self.metrics_collector.update_hourly_metrics(self.iteration_hour, current_flow, "power_consumed")
            self.engine.use_network_resources(current_flow)

            if self.dynamic_end_times:
                while self.active_flows and self.active_flows[0][0] <= self.current_step:
                    _, finished_flow = heapq.heappop(self.active_flows)
                    self.engine.release_network_resources(finished_flow)

                heapq.heappush(self.active_flows, (self.current_step + current_flow.duration, current_flow))

        done = not self.traffic
        flow = self.traffic.pop(0) if not done else None

        return (self.obs_builder.create_obs_dict(flow=flow),
                result.reward, done, False, self._make_info(current_flow))

    def _make_info(self, current_flow):
        key_start = f"flow_{current_flow.upf_location}_{current_flow.direction}"
        info_dict =  {
            f"{key_start}/granted": int(current_flow.is_granted()),
        }

        return info_dict | self.metrics_collector.return_info()

    def return_last_hour(self):
        return self.env_last_hour

    def return_hourly_results(self):
        return self.metrics_collector.return_hourly_results()

    def return_cpu_metrics(self):
        return self.metrics_collector.return_cpu_metrics()

    def return_metrics(self):
        return self.metrics_collector.return_info()

    def get_rehearsal_buffer(self):
        return self.rehearsal_buffer