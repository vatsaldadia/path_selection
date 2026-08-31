import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patches
import pickle

def draw_network(graph):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    color_map = {
        'DU': '#1f77b4', 'CU': '#ff7f0e',
        'UPF_TN': '#2ca02c', 'UPF_CN': '#d62728',
    }

    # 1. Extract positions from node attributes
    # We assume 'pos' attribute is a dict or list: [lat, long]
    pos = {}
    for node, data in graph.nodes(data=True):
        coords = data.get('pos', [0, 0])
        # Mapping: x = Longitude, y = Latitude
        # Adding a tiny amount of noise (1e-5) to prevent perfect overlap
        # np.random.seed(1)
        pos[node] = (coords[0] + np.random.uniform(-0.015, 0.015),
                     coords[1] + np.random.uniform(-0.015, 0.015))

    # 2. Dynamic Node Sizing
    # In lat/long, '0.1' is a massive distance.
    # We calculate node size based on the coordinate range.
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]
    range_x = max(x_values) - min(x_values)
    node_size = range_x * 0.025  # Boxes will be 5% of the total map width

    # Draw edges
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.15, edge_color='gray')

    for node in graph.nodes():
        x, y = pos[node]
        funcs = sorted(graph.nodes[node].get('functions', []))

        if not funcs:
            circle = patches.Circle((x, y), node_size / 2, linewidth=1, facecolor='lightgray', edgecolor='black',
                                    alpha=0.9, zorder=2)

            ax.add_patch(circle)

        else:

            label_color = 'black'
            label_weight = 'bold'

            # Draw Node Label
            # ax.text(x, y + (node_size * 0.8), str(node),
            #         fontsize=7, color=label_color, fontweight=label_weight,
            #         ha='center', va='center', clip_on=True)

            # Draw main box
            container = patches.Rectangle(
                (x - node_size / 2, y - node_size / 2),
                node_size, node_size,
                linewidth=1, edgecolor='black', facecolor='white', alpha=0.9, zorder=2
            )
            ax.add_patch(container)

            # Draw function stripes
            num_funcs = len(funcs)
            if num_funcs > 0:
                sub_width = node_size / num_funcs
                for i, func_name in enumerate(funcs):
                    sub_rect = patches.Rectangle(
                        (x - node_size / 2 + (i * sub_width), y - node_size / 2),
                        sub_width, node_size,
                        facecolor=color_map.get(func_name, 'gray'), edgecolor='white', lw=0.5, zorder=3
                    )
                    ax.add_patch(sub_rect)

        ax.text(x, y + (node_size * 0.8), str(node),
                fontsize=7, color=label_color, fontweight=label_weight,
                ha='center', va='center', clip_on=True)

        ax.axis('off')
    legend_elements = [patches.Patch(facecolor=c, label=n) for n, c in color_map.items()]
    fig.legend(handles=legend_elements, ncol=1, title="Functions")
    plt.savefig("network.png", dpi=300, bbox_inches='tight')
    plt.show()

with open("../init_modified.pkl", "rb") as f:
    loaded_data = pickle.load(f)

network: nx.Graph = loaded_data["network"]
for edge in network.edges():
    if (edge[1], edge[0]) in network.edges():
        network.remove_edge(edge[1], edge[0])
print(len(network.edges()))
print(network["TN_11"])
draw_network(network)