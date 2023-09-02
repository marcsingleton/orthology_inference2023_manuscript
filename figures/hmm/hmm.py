"""Make figure of alignments decoded with insertion HMM."""

import json
import os
import re

import homomorph
import matplotlib.pyplot as plt
import numpy as np
import skbio
from src.draw import plot_msa_data
from src.ortho_MSA import phylo
from src.utils import read_fasta

spid_regex = r'spid=([a-z]+)'

records = [('2252', 0, None),
           ('2A57', 0, 845),
           ('360E', 0, None)]
state_labels = ['1', '2', '3']
state_colors = ['C0', 'C1', 'C2']
k = 4

fig_width = 7.5
fig_height = 4.25
dpi = 600
plot_msa_kwargs = {'figsize': (fig_width, fig_height),
                   'left': 0.1, 'right': 0.92, 'top': 0.95, 'bottom': 0.04, 'anchor': (0, 0.5),
                   'height_ratio': 0.5, 'hspace': 0.4,
                   'x_labelsize': 5,
                   'data_min': -0.05, 'data_max': 1.05,
                   'data_labels': state_labels, 'data_linewidths': 1.5, 'data_colors': state_colors,
                   'tree_position': 0, 'tree_width': 0.1,
                   'tree_kwargs': {'linewidth': 0.5, 'tip_labels': False, 'xmin_pad': 0.01, 'xmax_pad': 0.025},
                   'msa_legend': True,
                   'legend_kwargs': {'bbox_to_anchor': (0.92, 0.5), 'loc': 'center left', 'fontsize': 6,
                                     'handletextpad': 0.5, 'markerscale': 1, 'handlelength': 1}}
panel_label = 'B'
panel_label_fontsize = 'large'
panel_label_offset = 0.025

tree_template = skbio.read('../../orthology_inference/analysis/ortho_tree/consensus_GTR2/out/NI.nwk', 'newick', skbio.TreeNode)
tree_order = skbio.read('../../orthology_inference/analysis/ortho_tree/consensus_LG/out/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_order.tips())}

with open('../../orthology_inference/analysis/ortho_MSA/insertion_hmm/out/model.json') as file:
    model_json = json.load(file)

if not os.path.exists('out/'):
    os.mkdir('out/')

for OGid, start, stop in records:
    # Load MSA
    msa = []
    for header, seq in read_fasta(f'../../orthology_inference/analysis/ortho_MSA/realign_fastas/out/{OGid}.afa'):
        spid = re.search(spid_regex, header).group(1)
        msa.append({'spid': spid, 'seq': seq})
    msa = sorted(msa, key=lambda x: tip_order[x['spid']])

    # Create emission sequence
    col0 = []
    emit_seq = []
    for j in range(len(msa[0]['seq'])):
        col = [1 if msa[i]['seq'][j] in ['-', '.'] else 0 for i in range(len(msa))]
        emit0 = sum([c0 == c for c0, c in zip(col0, col)])
        emit_seq.append(emit0)  # The tree probabilities are pre-calculated, so emission value is its index
        col0 = col
    emit_seq = np.array(emit_seq)

    # Load tree and convert to vectors at tips
    tree = tree_template.shear([record['spid'] for record in msa])
    for node in tree.postorder():  # Ensure tree is ordered as in original
        if node.is_tip():
            node.value = tip_order[node.name]
        else:
            node.children = sorted(node.children, key=lambda x: x.value)
            node.value = sum([child.value for child in node.children])

    tips = {tip.name: tip for tip in tree.tips()}
    for record in msa:
        spid, seq = record['spid'], record['seq']
        value = np.zeros((2, len(seq)))
        for j, sym in enumerate(seq):
            if sym in ['-', '.']:
                value[0, j] = 1
            else:
                value[1, j] = 1
        tip = tips[spid]
        tip.value = value

    # Instantiate model
    e_dists_rv = {}
    for s, e_dist in model_json['e_dists'].items():
        a, b, pinv, alpha, pi, q0, q1, p0, p1 = [e_dist[param] for param in ['a', 'b', 'pinv', 'alpha', 'pi', 'q0', 'q1', 'p0', 'p1']]
        pmf1 = phylo.get_betabinom_pmf(emit_seq, len(msa), a, b)
        pmf2 = phylo.get_tree_pmf(tree, pinv, k, alpha, pi, q0, q1, p0, p1)
        e_dists_rv[s] = phylo.ArrayRV(pmf1 * pmf2)
    model = homomorph.HMM(model_json['t_dists'], e_dists_rv, model_json['start_dist'])

    # Decode states and plot
    idx_seq = list(range(len(msa[0]['seq'])))  # Everything is pre-calculated, so emit_seq is the emit index
    fbs = model.forward_backward(idx_seq)
    data = [fbs[label] for label in state_labels]

    # Add additional height for SVG panel
    im = plt.imread('out/hmm_architecture.png')
    SVG_height = im.shape[0] / im.shape[1] * fig_width

    fig = plt.figure(figsize=(fig_width, SVG_height + fig_height))
    gs = plt.GridSpec(2, 1, height_ratios=[SVG_height, fig_height])

    subfig = fig.add_subfigure(gs[0])
    ax = subfig.add_axes((0, 0, 1, 1))
    ax.imshow(im)
    ax.axis('off')

    subfig = fig.add_subfigure(gs[1])
    plot_msa_data([record['seq'][start:stop] for record in msa], data[start:stop],
                  fig=subfig,
                  tree=tree, x_start=start, **plot_msa_kwargs)
    subfig.text(panel_label_offset / fig_width, 1 - panel_label_offset / fig_height, panel_label,
                fontsize=panel_label_fontsize, fontweight='bold',
                horizontalalignment='left', verticalalignment='top')
    plt.savefig(f'out/{OGid}.png', dpi=dpi)
    plt.savefig(f'out/{OGid}.tiff', dpi=dpi)
    plt.close()
