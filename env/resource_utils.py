from env.traffic_flow import TrafficFlow
from env.constants import NODE_TYPE_ATTRIBUTES
import time
import heapq


def get_processing_delay(flow: TrafficFlow):
    return sum(NODE_TYPE_ATTRIBUTES[func]["processing_time"]
               for func in ["RU", "DU", "CU", f"UPF_{flow.upf_location}"]) * flow.bandwidth * 0.0001

def get_cpu_utilization(node_data, node: str):
    return 1 - (node_data[node]["available"]["cpu"] / node_data[node]["capacity"]["cpu"])

def update_resources(node_data, edge_data, flow: TrafficFlow, action):
    if action == "use":
        multiplier =  -1
    elif action == "release":
        multiplier = 1
    else:
        raise ValueError(f"Unknown action: {action} while updating resources")

    for func, node in zip(["DU", "CU", f"UPF_{flow.upf_location}"], flow.chosen_datacenters):
        for k, v in NODE_TYPE_ATTRIBUTES[func]["resources"].items():
            node_data[node]["available"][k] += multiplier * v * flow.bandwidth * 0.01

    for u, v in zip(flow.chosen_path[:-1], flow.chosen_path[1:]):  # for edge resources
        edge_data[u][v]["available"] += multiplier * flow.bandwidth * 0.01