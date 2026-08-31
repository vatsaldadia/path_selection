import json

import networkx as nx
import numpy as np
import pickle
from dataclasses import dataclass

from env.constants import REWARD_SYSTEM
from env.power_utils import get_power_penalty, get_power_consumption_fan, get_idle_power_fan
from env.resource_utils import get_processing_delay, get_cpu_utilization, update_resources
from env.traffic_flow import TrafficFlow

@dataclass
class StepResult:
    reward: float = 0.0
    description: str | None = None

class Engine:
    def __init__(self, network_data, num_of_paths: int = 500):
        with open(f"rl_repo_{num_of_paths}_colocation_penalty.json") as f:
            self.path_sets_repo = json.load(f)
            self.action_space_dim = 0
            for i in self.path_sets_repo:
                for j in self.path_sets_repo[i]:
                    self.action_space_dim = max(self.action_space_dim, len(self.path_sets_repo[i][j]))

        self.network: nx.Graph = network_data["network"]
        self.datacenter_list : list[str] = sorted([node for node, data in self.network.nodes(data=True) if data["functions"]])
        self.ru_to_network: dict[str, str] = network_data["ru_to_network"]
        self.connector_list: list[str] = sorted(list(set(self.ru_to_network.values())))
        self.edge_data: dict[str, dict[str, dict[str, float]]] = network_data["edge_data"]
        self.node_data: dict[str, dict[str, dict[str, float]]] = network_data["node_data"]
        self.node_data_MASTER = network_data["node_data"]
        self.edge_data_MASTER = network_data["edge_data"]
        self.node_list: list[str] = sorted(list(self.network.nodes()))
        self.num_nodes: int = len(self.node_list)
        self.edge_list: list[tuple[str]] = sorted(list(self.network.edges()))
        self.num_edges: int = len(self.edge_list)

    def get_paths(self, flow: TrafficFlow):
        connector = self.ru_to_network[flow.ru]
        paths = [path_set[0] if flow.direction == "UL" else path_set[0][::-1]
                 for path_set in self.path_sets_repo[connector][flow.upf_location]]
        return paths

    def get_datacenters(self, flow: TrafficFlow):
        connector = self.ru_to_network[flow.ru]
        datacenters = [path_set[1] for path_set in self.path_sets_repo[connector][flow.upf_location]]
        return datacenters

    def _mask_check_step(self, flow: TrafficFlow, result: StepResult):
        result.reward += REWARD_SYSTEM["failure"]

        processing_delay = get_processing_delay(flow)
        if processing_delay > flow.latency:
            result.description = "processing_delay_error"                        # check to see if any flow has more processing delay than latency

        for path_idx, datacenters in enumerate(self.get_datacenters(flow)):                 # capacity threshold check
            for datacenter in datacenters:
                if self.get_datacenter_cpu_utilization(datacenter) > 0.8:
                    result.description = "datacenter_cpu_dead_end"
                    break

        for path_idx, path in enumerate(self.get_paths(flow)):
            propagation_delay = 0
            for u, v in zip(path[:-1], path[1:]):
                propagation_delay += self.edge_data[u][v]["delay"]
                if self.edge_data[u][v]["available"] < flow.bandwidth * 0.01:       # bandwidth threshold check
                    result.description = "bandwidth_dead_end"
                    break

            if processing_delay + propagation_delay > flow.latency:                 # latency threshold check
                result.description = "latency_dead_end"
                break

            if result.description:
                break

        return result

    def make_step(self, flow: TrafficFlow, action: int, mask: np.ndarray):
        result = StepResult()

        # if all(m == 0 for m in mask):  # check if there are no legal moves due to mask
        #     return self._mask_check_step(flow, result)

        processing_delay = get_processing_delay(flow)
        if processing_delay > flow.latency:
            result.description = "processing_delay_error"  # check to see if any flow has more processing delay than latency
            result.reward += REWARD_SYSTEM["failure"]
            return result

        # capacity threshold check
        for datacenter in self.get_datacenters(flow)[action]:
            if self.get_datacenter_cpu_utilization(datacenter) > 0.8:
                result.description = "datacenter_cpu_dead_end"
                result.reward += REWARD_SYSTEM["failure"]
                return result

        path = self.get_paths(flow)[action]
        propagation_delay = 0
        for u, v in zip(path[:-1], path[1:]):
            propagation_delay += self.edge_data[u][v]["delay"]
            if self.edge_data[u][v]["available"] < flow.bandwidth * 0.01:  # bandwidth threshold check
                result.description = "bandwidth_dead_end"
                result.reward += REWARD_SYSTEM["failure"]
                return result

            if processing_delay + propagation_delay > flow.latency:  # latency threshold check
                result.description = "latency_dead_end"
                result.reward += REWARD_SYSTEM["failure"]
                return result

            if result.description:
                result.reward += REWARD_SYSTEM["failure"]
                return result

        if not mask[action]:
            raise ValueError(
                f"Illegal action: {action}, path {self.get_paths(flow)[action]} is not a valid action when connector is "
                f"{self.ru_to_network[flow.ru]}, upf location is {flow.upf_location}, "
                f"and direction is {flow.direction}")

        flow.chosen_path = self.get_paths(flow)[action]
        flow.chosen_datacenters = self.get_datacenters(flow)[action]
        flow.grant()

        result.reward += REWARD_SYSTEM["success"]
        min_avail_cpu = min(1 - self.get_datacenter_cpu_utilization(dc) for dc in flow.chosen_datacenters)
        normalized_penalty = (min_avail_cpu - 0.2) / 0.8           # this is based on the lines of 80% cpu capacity
        result.reward += REWARD_SYSTEM["scaling_alpha"] * np.log(normalized_penalty + 0.0001)
        result.description = "granted"

        return result

    def get_datacenter_cpu_utilization(self, node: str):
        return get_cpu_utilization(self.node_data, node)

    def get_power(self, flow: TrafficFlow):
        return get_power_consumption_fan(self.node_data, flow)

    def get_idle_power(self):
        return get_idle_power_fan(self.node_list)

    def use_network_resources(self, flow: TrafficFlow):
        update_resources(self.node_data, self.edge_data, flow, "use")

    def release_network_resources(self, flow: TrafficFlow):
        update_resources(self.node_data, self.edge_data, flow, "release")

    def reset_network_resources(self):
        with open("./init_modified.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        self.edge_data: dict[str, dict[str, dict[str, float]]] = loaded_data["edge_data"]
        self.node_data: dict[str, dict[str, dict[str, float]]] = loaded_data["node_data"]
        for node, val in self.node_data.items():
            for k, v in val["capacity"].items():
                val["capacity"][k] *= 1.0
            val["available"] = val["capacity"].copy()
        for u, edge in self.edge_data.items():
            for v, data in edge.items():
                data["capacity"] *= 0.5
                data["available"] = data["capacity"]