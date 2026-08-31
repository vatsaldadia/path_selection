import pickle
from collections import defaultdict
import networkx as nx

with open("./init_state_old.pkl", "rb") as f:
    loaded_data = pickle.load(f)

ru_to_du = loaded_data["ru_to_du"]
du_to_cu = loaded_data["du_to_cu"]
upf_tn_roles = loaded_data["cu_to_upf_tn"]
upf_cn_roles = loaded_data["cu_to_upf_cn"]

capacities = loaded_data["capacities"]
for k, v in capacities.copy().items():  # change node features to lower case
    if 0 not in v.values():  # ignore nodes with 0 capacity
        for k_ in v.copy():
            capacities[k][k_.lower()] = capacities[k][k_]
            del capacities[k][k_]
    else:
        del capacities[k]

for ru in ru_to_du.copy():
    ru_to_du[ru[:-2]] = ru_to_du[ru]
    del ru_to_du[ru]

upf_by_location = {"UPF_TN": set(), "UPF_CN": set()}
for i in upf_tn_roles:
    upf_by_location["UPF_TN"].update(set(upf_tn_roles[i].values()))
for i in upf_cn_roles:
    upf_by_location["UPF_CN"].update(set(upf_cn_roles[i].values()))

upf_to_cu = defaultdict(list)  # map UPF to directly linked CUs
cus_from_upf = set()  # set of CUs that are reachable from UPFs
for upf_type, upf_map in upf_tn_roles.items():
    for cu, upf in upf_map.items():
        upf_to_cu[upf].append(cu)
        cus_from_upf.add(cu)
for upf_type, upf_map in upf_cn_roles.items():
    for cu, upf in upf_map.items():
        upf_to_cu[upf].append(cu)
        cus_from_upf.add(cu)

all_cus = list(set(du_to_cu.values()).intersection(cus_from_upf))  # get all CUs linked to DUs AND UPFs

node_types = defaultdict(set)

for ru, du in ru_to_du.items():
    node_types[du].add("DU")

for cu in all_cus:
    node_types[cu].add("CU")

for upf_location, upf in upf_by_location.items():
    for upf_node in upf:
        node_types[upf_node].add(upf_location)

network = loaded_data["G"]
network.remove_node("DN")

edges_cn8 = [
    ("TN_12", "CN_12", {"delay": 0.2, "bandwidth": network["CN_8"]["CN_12"]["bandwidth"]}),
    ("TN_17", "CN_4", {"delay": 0.2, "bandwidth": network["CN_8"]["CN_4"]["bandwidth"]}),
    ("TN_17", "CN_2", {"delay": 0.2, "bandwidth": network["CN_8"]["CN_2"]["bandwidth"]})
]
reversed_edges_cn8 = [
    ("CN_12", "TN_12", {"delay": 0.2, "bandwidth": network["CN_12"]["CN_8"]["bandwidth"]}),
    ("CN_4", "TN_17", {"delay": 0.2, "bandwidth": network["CN_4"]["CN_8"]["bandwidth"]}),
    ("CN_2", "TN_17", {"delay": 0.2, "bandwidth": network["CN_2"]["CN_8"]["bandwidth"]})
]

edges_cn9 = [
    ("TN_23", "CN_6", {"delay": 0.2, "bandwidth": network["CN_9"]["CN_6"]["bandwidth"]}),
    ("TN_23", "CN_14", {"delay": 0.2, "bandwidth": network["CN_9"]["CN_14"]["bandwidth"]}),
    ("TN_23", "CN_1", {"delay": 0.2, "bandwidth": network["CN_9"]["CN_1"]["bandwidth"]})
]
reversed_edges_cn9 = [
    ("CN_6", "TN_23", {"delay": 0.2, "bandwidth": network["CN_6"]["CN_9"]["bandwidth"]}),
    ("CN_14", "TN_23", {"delay": 0.2, "bandwidth": network["CN_14"]["CN_9"]["bandwidth"]}),
    ("CN_1", "TN_23", {"delay": 0.2, "bandwidth": network["CN_1"]["CN_9"]["bandwidth"]})
]

network.remove_node("CN_8")
network.remove_node("CN_9")
network.add_edges_from(edges_cn8)
network.add_edges_from(reversed_edges_cn8)
network.add_edges_from(edges_cn9)
network.add_edges_from(reversed_edges_cn9)

distances = {}
for u, v in network.edges():
    pos_u = (network.nodes[u]['pos'][1], network.nodes[u]['pos'][0])
    pos_v = (network.nodes[v]['pos'][1], network.nodes[v]['pos'][0])
    dist = geodesic(pos_u, pos_v).meters
    distances[(u, v)] = dist
max_dist = max(distances.values())
min_dist = min(distances.values())
for (u, v), dist in distances.items():
    # Longer distance -> higher latency (closer to 0.3)
    normalized = (dist - min_dist) / (max_dist - min_dist)
    score = 0.1 + (normalized * (0.3 - 0.1))
    network[u][v]['delay'] = round(score, 4)
    print(f"{u} -> {v}: {score}")

edge_data = defaultdict(dict)
for u, v, data in network.edges(data=True):
    temp_data = {"delay": data["delay"], "capacity": data["bandwidth"], "available": data["bandwidth"]}
    edge_data[u][v] = temp_data.copy()

ru_to_network = dict()
for node in network:
    if node.startswith("RU"):
        ru_to_network[node] = [n for n in network[node]][0]

for node in network:
    if "capacity" in network.nodes[node]:
        del network.nodes[node]["capacity"]
    if "type" in network.nodes[node]:
        del network.nodes[node]["type"]
node_type_attributes = defaultdict(dict)
for node, types in node_types.items():
    node_type_attributes[node] = {"functions": set(types)}
for node in network.nodes():
    if node not in node_type_attributes:
        node_type_attributes[node] = {"functions": set()}
nx.set_node_attributes(network, node_type_attributes)

network_non_ru = network.copy()
network_non_ru.remove_nodes_from([node for node in network.nodes() if node.startswith("RU")])

node_data = defaultdict(dict)
for n in network_non_ru.nodes():
    if n in capacities:
        node_data[n] = {"capacity": capacities.get(n).copy(), "available": capacities.get(n).copy()}

init_modified = {
    "network_with_ru": network,
    "network": network_non_ru,
    "edge_data": edge_data,
    "node_data": node_data,
    "ru_to_du": ru_to_du,
    "ru_to_network": ru_to_network
}

# print(len(network_non_ru.nodes()))

with open('./init_modified.pkl', 'wb') as pkl_file:
    pickle.dump(init_modified, pkl_file)