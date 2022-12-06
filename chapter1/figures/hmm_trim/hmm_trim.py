"""Make figure of HMM trimming data."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import SubplotParams
from matplotlib.gridspec import GridSpec

# Load OGids
OGids = []
with open('../../orthology_inference/analysis/ortho_MSA/realign_fastas/out/errors.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, error_flag1, error_flag2 = fields['OGid'], fields['error_flag1'], fields['error_flag2']
        if error_flag1 == 'False' and error_flag2 == 'False':
            OGids.append(OGid)

df = pd.read_table('../../orthology_inference/analysis/ortho_MSA/insertion_trim/out/trim_stats.tsv')

df['length'] = df['stop'] - df['start']
df['length_ratio'] = df['length'] / df['colnum']
df['norm2'] = df['posterior2'] / df['length']
df['norm3'] = df['posterior3'] / df['length']
groups = df.groupby('OGid')

if not os.path.exists('out/'):
    os.mkdir('out')

fig = plt.figure(figsize=(7.5, 4.5), subplotpars=SubplotParams(0.075, 0.075, 0.925, 0.925))
gs = GridSpec(2, 2)

# Pie chart by presence of trims
values = [len(set(OGids) - set(df['OGid'])), len(set(df['OGid']))]
labels = [f'{label}\n{value:,}' for label, value in zip(['w/o trims', 'w/ trims'], values)]
subfig = fig.add_subfigure(gs[0, 0], facecolor='none')
ax = subfig.add_subplot()
ax.pie(values, labels=labels, labeldistance=1.15)
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# Distribution of length ratio of total trims
subfig = fig.add_subfigure(gs[:, 1], facecolor='none')
subgs = subfig.add_gridspec(4, 2, height_ratios=(1, 2, 0.15, 0.1), width_ratios=(4, 1))
ax = subfig.add_subplot(subgs[1, 0])
ax_histx = subfig.add_subplot(subgs[0, 0], sharex=ax)
ax_histy = subfig.add_subplot(subgs[1, 1], sharey=ax)

hb = ax.hexbin(groups.size(), groups['length_ratio'].sum(), bins='log', gridsize=50, mincnt=1, linewidth=0)
cax = subfig.add_subplot(subgs[3, 0])
subfig.colorbar(hb, cax=cax, orientation='horizontal')

counts = groups.size().value_counts()
ax_histx.bar(counts.index, counts.values, width=1)
ax_histy.hist(groups['length_ratio'].sum(), bins=100, orientation='horizontal')

ax.set_xlabel('Number of trims in OG')
ax.set_ylabel('Total length\nratio of trims in OG')
subfig.suptitle('B', x=0.025, y=0.9875, fontweight='bold')  # In figure coordinates, so half distance of others

# Hexbin of posterior2 vs posterior3
subfig = fig.add_subfigure(gs[1, 0], facecolor='none')
ax = subfig.add_axes((0.15, 0.275, 0.7, 0.7), aspect='equal')
hb = ax.hexbin(df['norm2'], df['norm3'], bins='log', gridsize=25, mincnt=1, linewidth=0)
ax.set_xlabel('Average state 2\nposterior in trim')
ax.set_ylabel('Average state 3\nposterior in trim')
fig.colorbar(hb)
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/hmm_trim.png', dpi=300)
fig.savefig('out/hmm_trim.tiff', dpi=300)
plt.close()
