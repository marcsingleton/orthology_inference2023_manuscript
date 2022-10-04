"""Make figure of species trees."""

import os

import matplotlib.pyplot as plt
import skbio
from matplotlib import rcParams
from src.draw import plot_tree

if not os.path.exists('out/'):
    os.mkdir('out/')

# FIGURE WITH TWO TREES
trees = []
for model_label in ['LG', 'GTR']:
    tree = skbio.read('../../IDREvoDevo/analysis/ortho_tree/consensus_LG/out/100R_NI.nwk', 'newick', skbio.TreeNode)
    tree.assign_supports()
    for node in tree.traverse():
        if node.support == 1:
            node.support = None  # Clear supports of 1 to simplify plot
    trees.append(tree)

xlabels = ['amino acid', 'nucleotide']
title_labels = ['A', 'B']
dpi = 300

fig, axs = plt.subplots(1, 2, figsize=(7.5, 4), layout='tight')
for ax, tree, xlabel, title_label in zip(axs, trees, xlabels, title_labels):
    plot_tree(tree, ax, tip_fontsize=8, support_labels=True,
              support_format_spec='.2f', support_fontsize=8.5,
              support_ha='right', support_hoffset=-0.006, support_voffset=-0.006)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(f'{xlabel} substitution per site')
    ax.set_title(title_label, loc='left', fontdict={'fontweight': 'bold'})

plt.savefig('out/trees1.png', dpi=dpi)
plt.savefig('out/trees1.tiff', dpi=dpi)
plt.close()

# SUPPLEMENTS WITH DIFFERENT SAMPLING STRATEGIES
file_labels = ['0R', '50R', '100R',
               '0R_NI', '50R_NI', '100R_NI']
model_labels = ['LG', 'GTR']
xlabels = ['amino acid', 'nucleotide']

ylabels = ['invariant', 'non-invariant']
title_labels = ['0% redundancy', '50% redundancy', '100% redundancy']

for model_label, xlabel in zip(model_labels, xlabels):
    trees = []
    for file_label in file_labels:
        tree = skbio.read(f'../../IDREvoDevo/analysis/ortho_tree/consensus_{model_label}/out/{file_label}.nwk', 'newick', skbio.TreeNode)
        tree.assign_supports()
        for node in tree.traverse():
            if node.support == 1:
                node.support = None  # Clear supports of 1 to simplify plot
        trees.append(tree)

    fig, axs = plt.subplots(len(ylabels), len(title_labels), figsize=(7.5, 6), layout='constrained')
    for ax, tree in zip(axs.ravel(), trees):
        plot_tree(tree, ax, tip_fontsize=6, tip_offset=0.002, support_labels=True,
                  support_format_spec='.2f', support_fontsize=6,
                  support_ha='right', support_hoffset=-0.007, support_voffset=-0.007)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('')

    for ax in axs[1]:
        ax.set_xlabel(f'{xlabel} substitutions per site')
    for ax, ylabel in zip(axs[:, 0], ylabels):
        ax.set_ylabel(ylabel, fontsize=rcParams['axes.titlesize'])
    for ax, title_label in zip(axs[0], title_labels):
        ax.set_title(title_label)

    fig.savefig(f'out/trees2_{model_label}.png', dpi=dpi)
    fig.savefig(f'out/trees2_{model_label}.tiff', dpi=dpi)
    plt.close()
