"""Make figure of alignments before and after alignment with HMMer."""

import os
import re

import matplotlib.pyplot as plt
import skbio
from src.draw import plot_msa
from src.utils import read_fasta

OGids = ['2AB6', '07C3', '31B4', '269C', '06D1', '23D9']
msa_pathA = '../../IDREvoDevo/analysis/ortho_MSA/get_repseqs/out/'
msa_pathB = '../../IDREvoDevo/analysis/ortho_MSA/realign_hmmer/out/mafft/'
spid_regex = r'spid=([a-z]+)'

adjust_left = 0.01
adjust_bottom = 0.075
adjust_right = 0.915
adjust_top = 0.925

hspace = 0.3
x_labelsize = 5
legend_position = (0.915, 0.5)
legend_fontsize = 6
legend_handletextpad = 0.5
legend_markerscale = 1

fig_width = 7.5
fig_height = 3
figsize_effective = (fig_width * (adjust_right - adjust_left), fig_height * (adjust_top - adjust_bottom))
dpi = 300

panel_labels = ['A', 'B']
panel_label_fontsize = 'large'
panel_label_offset = 0.025

tree_order = skbio.read('../../IDREvoDevo/analysis/ortho_tree/consensus_LG/out/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_order.tips())}

if not os.path.exists('out/'):
    os.mkdir('out/')

for OGid in OGids:
    for panel_label in panel_labels:
        if panel_label == 'A':
            path = f'{msa_pathA}/{OGid}.afa'
        else:
            path = f'{msa_pathB}/{OGid}.afa'

        msa = []
        for header, seq in read_fasta(path):
            spid = re.search(spid_regex, header).group(1)
            msa.append({'spid': spid, 'seq': seq})
        msa = sorted(msa, key=lambda x: tip_order[x['spid']])

        fig = plot_msa([record['seq'] for record in msa],
                       figsize=figsize_effective, hspace=hspace,
                       x_labelsize=x_labelsize,
                       msa_legend=True, legend_kwargs={'bbox_to_anchor': legend_position, 'loc': 'center left', 'fontsize': legend_fontsize,
                                                       'handletextpad': legend_handletextpad, 'markerscale': legend_markerscale, 'handlelength': 1})
        fig.text(panel_label_offset / fig_height, 1 - panel_label_offset / fig_width, panel_label, fontsize=panel_label_fontsize, fontweight='bold',
                 horizontalalignment='left', verticalalignment='top')
        plt.subplots_adjust(left=adjust_left, bottom=adjust_bottom, right=adjust_right, top=adjust_top)
        plt.savefig(f'out/{OGid}_{panel_label}.png', dpi=dpi)
        plt.savefig(f'out/{OGid}_{panel_label}.tiff', dpi=dpi)
        plt.close()
