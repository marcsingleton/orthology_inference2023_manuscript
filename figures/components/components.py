"""Make figure of components."""

import os
from math import floor, log10

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec
from numpy import linspace

component_id_groups = [('000C', '001E'),
                       ('0030', '0031'),
                       ('0022', '002F'),
                       ('000F', '0013'),
                       ('0007', '0004'),
                       ('002B', '002F')]

fig_width = 7.5
fig_height = 3.75
dpi = 300

margin_width = 0
margin_height = 0
margin_data = 0.075

panel_labels = ['A', 'B']
panel_label_fontsize = 'large'
panel_label_offset = 0.025

node_color2 = '#444444'
node_slope2 = 200
node_intercept2 = 0
edge_alpha2 = 0.3
edge_width2 = 0.75

cmap_base = mpl.cm.get_cmap('plasma_r', 320)  # 64 additional levels so truncated cmap has 256
cmap = mpl.colors.ListedColormap(cmap_base(linspace(0.25, 1, 256)))
cbar_fontsize = 8
cbar_width = 0.125
cbar_height = 0.02
cbar_offset_right = 0.01
cbar_offset_bottom = 0.075

# Load graph
graph = {}
with open('../../orthology_inference/analysis/ortho_cluster2/hits2graph/out/hit_graph.tsv') as file:
    for line in file:
        node, adjs = line.rstrip('\n').split('\t')
        graph[node] = [adj.split(':') for adj in adjs.split(',')]

# Load connected components
components = {}
with open('../../orthology_inference/analysis/ortho_cluster2/connect_hit_graph/out/components.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        components[fields['component_id']] = set(fields['ppids'].split(','))

if not os.path.exists('out/'):
    os.mkdir('out/')

for component_ids in component_id_groups:
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(1, len(component_ids), left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)
    panel_width = fig_width * (rights[0] - lefts[0])
    panel_height = fig_height * (tops[0] - bottoms[0])

    for i, (component_id, panel_label) in enumerate(zip(component_ids, panel_labels)):
        # Create graph
        component = components[component_id]
        subgraph = {node: graph[node] for node in component}

        nx_graph = nx.Graph()
        for node, adjs in sorted(subgraph.items()):
            nx_graph.add_node(node)
            for adj, w in adjs:
                if (node, adj) in nx_graph.edges:
                    nx_graph.edges[node, adj]['weight'] += float(w)
                else:
                    nx_graph.add_edge(node, adj, weight=float(w))

        # Get positions and axes limits
        positions = nx.kamada_kawai_layout(nx_graph, weight=None)  # Weight is None as otherwise is used for layout
        xs = [xy[0] for xy in positions.values()]
        xmin, xmax = min(xs), max(xs)
        xlen = xmax - xmin
        ys = [xy[1] for xy in positions.values()]
        ymin, ymax = min(ys), max(ys)
        ylen = ymax - ymin

        # Rotate positions
        if xlen / ylen > 1 and panel_width / panel_height < 1:  # Make long sides match
            xmin, xmax, ymin, ymax = ymin, ymax, -xmax, -xmin
            xlen, ylen = ylen, xlen
            positions = {node: (y, -x) for node, (x, y) in positions.items()}

        # Fit axes into panel
        if xlen / ylen > panel_width / panel_height:  # Axes is wider than panel
            width = panel_width / fig_width
            height = panel_width / fig_height * ylen / xlen
        else:
            height = panel_height / fig_height
            width = panel_height / fig_width * xlen / ylen
        node_size2 = node_slope2 * panel_width * panel_height / len(subgraph) + node_intercept2  # Node size is inversely proportional to node density

        # Draw graph labeled by edge
        edges = sorted(nx_graph.edges, key=lambda x: nx_graph.edges[x]['weight'])
        ws = [w for _, _, w in nx_graph.edges.data('weight')]
        wmin, wmax = min(ws), max(ws)
        wlen = wmax - wmin

        ax = fig.add_subplot(gs[0, i], aspect='equal')
        x0, y0, x1, y1 = gs[0, i].get_position(fig).extents

        nx.draw_networkx_edges(nx_graph, positions, ax=ax, edgelist=edges, alpha=edge_alpha2, width=edge_width2, edge_color=ws, edge_cmap=cmap)
        nx.draw_networkx_nodes(nx_graph, positions, ax=ax, node_size=node_size2, linewidths=0, node_color=node_color2)
        ax.set_xlim((xmin - margin_data * xlen, xmax + margin_data * xlen))  # Set manually because draw_networkx_edges hard codes the data limits with 5% padding
        ax.set_ylim((ymin - margin_data * ylen, ymax + margin_data * ylen))

        cax = fig.add_axes((x0 + panel_width / fig_width - cbar_width - cbar_offset_right, cbar_offset_bottom,
                            cbar_width, cbar_height))
        ticks = [wmin + wlen / 4, wmax - wlen / 4]
        ticklabels = [f'{round(tick, -floor(log10(tick)) + 2):5g}' for tick in ticks]  # Round to third significant digit of difference between ticks
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(wmin, wmax), cmap=cmap), cax=cax, orientation='horizontal')
        cbar.ax.set_xticks(ticks, ticklabels, fontsize=cbar_fontsize)

        fig.text(x0 + panel_label_offset / fig_width, 1 - panel_label_offset / fig_height, panel_label, fontsize=panel_label_fontsize, fontweight='bold',
                 horizontalalignment='left', verticalalignment='top')
        ax.axis('off')
    fig.savefig(f'out/{component_ids[0]}-{component_ids[1]}.png', dpi=dpi)
    fig.savefig(f'out/{component_ids[0]}-{component_ids[1]}.tiff', dpi=dpi)
    plt.close()
