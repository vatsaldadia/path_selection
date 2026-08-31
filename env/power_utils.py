from env.traffic_flow import TrafficFlow
from env.constants import NODE_TYPE_ATTRIBUTES, POWER_VALUES


def get_power_penalty(node_data, node: str):
    return ((-1 / node_data[node]["capacity"]["cpu"]) * 1e4) / 3            # scale down penalty as choosing 3 targets


def get_power_consumption_fan(node_data, flow: TrafficFlow):  # don't include idle power of nodes
    """
    Calculate the power consumption of the current flow.
    """
    total_power = 0.0

    for func, node in zip(["DU", "CU", f"UPF_{flow.upf_location}"], flow.chosen_datacenters):
        power_params = NODE_TYPE_ATTRIBUTES[func]["power_params"]
        function_cpu_demand = (NODE_TYPE_ATTRIBUTES[func]["resources"]["cpu"] *
                               flow.bandwidth * 0.01)
        cpu_capacity = node_data[node]["capacity"]["cpu"]

        total_power += ((power_params["max"] - power_params["idle"]) *
                        (function_cpu_demand / cpu_capacity))

    power_params = NODE_TYPE_ATTRIBUTES["switch"]["power_params"]
    total_power += power_params["bit"] * flow.bandwidth * len(flow.chosen_path)

    return total_power


def get_idle_power_fan(node_list):
    """
    Calculate the idle power consumption of all nodes.
    """
    total_idle_power = 0.0

    for node in node_list:
        max_idle_power = 0.0
        for node_type in node["functions"]:
            max_idle_power = max(max_idle_power, NODE_TYPE_ATTRIBUTES[node_type]["power_params"]["idle"])
        total_idle_power += max_idle_power  # take max idle power of all the functions the node has
    else:
        total_idle_power += NODE_TYPE_ATTRIBUTES["switch"]["power_params"]["idle"]

    return total_idle_power


def get_power_consumption_seb(node_data, state: TrafficFlow):
    total_power = 0.0
    for function, node in state.chosen_datacenters.items():
        if function != "CONNECTOR":
            function_cpu_demand = (NODE_TYPE_ATTRIBUTES[function]["resources"]["cpu"]
                                   * state.flow["bandwidth"] * 0.01)
            cpu_capacity = node_data[node]["capacity"]["cpu"]
            cpu_util = function_cpu_demand / cpu_capacity
            p_cpu = ((POWER_VALUES["alpha"]["a1"] * cpu_util
                      + POWER_VALUES["alpha"]["a2"] * (cpu_util ** 2)
                      + POWER_VALUES["alpha"]["a3"] * (cpu_util ** 3)
                      + POWER_VALUES["alpha"]["a4"] * (cpu_util ** 4))
                     * POWER_VALUES["freq_Hz"]) * POWER_VALUES["w_W_per_Hz"]

            function_mem_demand = (NODE_TYPE_ATTRIBUTES[function]["resources"]["memory"]
                                   * state.flow["bandwidth"] * 0.01)
            mem_capacity = node_data[node]["capacity"]["memory"]
            mem_util = function_mem_demand / mem_capacity
            p_mem = POWER_VALUES["alpha"]["a6"] * mem_util

            function_storage_demand = (NODE_TYPE_ATTRIBUTES[function]["resources"]["storage"]
                                       * state.flow["bandwidth"] * 0.01)
            storage_capacity = node_data[node]["capacity"]["storage"]
            storage_util = function_storage_demand / storage_capacity
            p_storage = POWER_VALUES["alpha"]["a7"] * storage_util

            total_power += p_cpu + p_mem + p_storage

    power_params = NODE_TYPE_ATTRIBUTES["switch"]["power_params"]
    total_power += power_params["bit"] * state.flow["total_bandwidth"] * len(state.path)

    return total_power


def get_idle_power_seb():                               # TODO: complete the function
    pass