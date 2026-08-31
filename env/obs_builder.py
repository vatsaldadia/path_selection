from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
from gymnasium import spaces

from env.engine import Engine
from env.resource_utils import get_processing_delay
from env.traffic_flow import TrafficFlow

class ObservationBuilder:

    def __init__(self, engine):
        self.engine = engine
        self.current_flow: Optional[TrafficFlow] = None

        self.observation_space = spaces.Dict({
            "connector": spaces.Discrete(len(self.engine.connector_list)),
            "upf_location": spaces.MultiBinary(1),              # 0 is TN, 1 is CN
            "direction": spaces.MultiBinary(1),                # 0 is UL, 1 is DL
            "datacenters": spaces.Box(low=0, high=(len(self.engine.datacenter_list) - 1), shape=(self.engine.action_space_dim, 3), dtype=np.int32),
            "datacenters_available_cpu": spaces.Box(low=0, high=1, shape=(self.engine.action_space_dim, 3), dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.engine.action_space_dim),
        })

        self.obs = {
            key: np.zeros(space.shape, dtype=space.dtype)
            if hasattr(space, 'shape') else 0
            for key, space in self.observation_space.spaces.items()
        }

    def create_obs_dict(self, flow: Optional[dict]):
        """
        Populate the `self.obs` dictionary with the current flow and return it.
        This is the observation that the policy will receive.
        """
        if flow is None:                                                    # dummy obs when episode is done
            return {
                key: np.zeros(space.shape, dtype=space.dtype)
                if hasattr(space, 'shape') else 0
                for key, space in self.observation_space.spaces.items()
            }

        self.current_flow = TrafficFlow(flow)
        assert self.current_flow is not None

        self.obs["connector"] = self.engine.connector_list.index(self.engine.ru_to_network[self.current_flow.ru])
        self.obs["upf_location"] = np.array([1 if self.current_flow.upf_location == "CN" else 0], dtype=np.int8)
        self.obs["direction"] = np.array([1 if self.current_flow.direction == "DL" else 0], dtype=np.int8)

        datacenters_np = np.zeros_like(self.obs["datacenters"])
        for path_idx, datacenters in enumerate(self.engine.get_datacenters(self.current_flow)):
            for dc_idx, datacenter in enumerate(datacenters):
                node_idx = self.engine.datacenter_list.index(datacenter)
                datacenters_np[path_idx, dc_idx] = node_idx
        self.obs["datacenters"] = datacenters_np

        datacenters_available_cpu = np.zeros_like(self.obs["datacenters_available_cpu"])
        for path_idx, datacenters in enumerate(self.engine.get_datacenters(self.current_flow)):
            for dc_idx, datacenter in enumerate(datacenters):
                available_cpu = 1 - self.engine.get_datacenter_cpu_utilization(datacenter)
                datacenters_available_cpu[path_idx, dc_idx] = available_cpu
        self.obs["datacenters_available_cpu"] = datacenters_available_cpu

        self.obs["action_mask"] = self._get_action_mask_array().astype(np.int8)

        return self.obs

    def _get_action_mask_array(self):
        """
        Returns a boolean mask of legal actions (node indices) for the current flow.
        This mask applies to ALL N nodes in the graph.
        """
        mask = np.zeros(self.engine.action_space_dim, dtype=bool)
        for idx in range(len(self.engine.get_paths(self.current_flow))):
            mask[idx] = True

        # assert self.current_flow is not None
        # processing_delay = get_processing_delay(self.current_flow)
        # if processing_delay > self.current_flow.latency:                 # check to see if any flow has more processing delay than latency
        #     return mask
        #
        # for path_idx, (path, datacenters) in enumerate(
        #         zip(self.engine.get_paths(self.current_flow), self.engine.get_datacenters(self.current_flow))):
        #     is_valid = True
        #
        #     for datacenter in datacenters:                                               # capacity threshold check
        #         if self.engine.get_datacenter_cpu_utilization(datacenter) > 0.8:
        #             is_valid = False
        #             break
        #     if not is_valid:
        #         continue
        #
        #     propagation_delay = 0
        #     for u, v in zip(path[:-1], path[1:]):
        #         if self.engine.edge_data[u][v]["available"] < self.current_flow.bandwidth * 0.01:       # bandwidth threshold check
        #             is_valid = False
        #             break
        #         propagation_delay += self.engine.edge_data[u][v]["delay"]
        #     if not is_valid:
        #         continue
        #
        #     if processing_delay + propagation_delay > self.current_flow.latency:                 # latency threshold check
        #         is_valid = False
        #
        #     if is_valid:
        #         mask[path_idx] = True

        return mask