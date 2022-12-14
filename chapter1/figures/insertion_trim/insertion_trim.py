"""Make figure of insertion HMM trimming data."""

import os

import matplotlib.pyplot as plt
import numpy as np
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

df1 = pd.read_table('../../orthology_inference/analysis/ortho_MSA/insertion_trim/out/seq_stats.tsv')
df2 = pd.read_table('../../orthology_inference/analysis/ortho_MSA/insertion_trim/out/region_stats.tsv')

df2['length'] = df2['stop'] - df2['start']
df2['length_ratio'] = df2['length'] / df2['colnum']
df2['norm2'] = df2['posterior2'] / df2['length']
df2['norm3'] = df2['posterior3'] / df2['length']

groups1 = df1.groupby('OGid')
groups2 = df2.groupby('OGid')

if not os.path.exists('out/'):
    os.mkdir('out')

fig = plt.figure(figsize=(7.5, 8), subplotpars=SubplotParams(0.25, 0.325, 0.9, 0.925))
gs = GridSpec(4, 2)

# Pie chart by presence of trims
union = set(df1['OGid']) | set(df2['OGid'])
intersection = set(df1['OGid']) & set(df2['OGid'])
OGids1 = set(df1['OGid']) - intersection
OGids2 = set(df2['OGid']) - intersection

values = [len(set(OGids)) - len(union), len(OGids1), len(OGids2), len(intersection)]
labels = [f'{label}\n({value:,})' for label, value in zip(['no trims', 'segment trims only', 'region trims only', 'segment and region trims'], values)]
colors = ['C0', 'C2', 'C1', 'C3']
subfig = fig.add_subfigure(gs[0, 0], facecolor='none')
ax = subfig.add_axes((0, 0.1, 0.45, 0.9))
ax.pie(values, labels=labels, labeldistance=None, colors=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# Hexbin of posterior2 vs posterior3 in region trims
subfig = fig.add_subfigure(gs[0, 1], facecolor='none')
ax = subfig.add_subplot()
hb = ax.hexbin(df2['norm2'], df2['norm3'], bins='log', gridsize=25, mincnt=1, linewidth=0)
ax.set_xlabel('Average state 2\nposterior in region trim')
ax.set_ylabel('Average state 3\nposterior in region trim')
fig.colorbar(hb)
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

# Distribution of number of segment trims in OGs
counts = groups1.size().value_counts()
subfig = fig.add_subfigure(gs[1, 0], facecolor='none')
ax = subfig.add_subplot()
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of segment trims in OG')
ax.set_ylabel('Number of OGs')
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')

# Distribution of number of region trims in OGs
counts = groups2.size().value_counts()
subfig = fig.add_subfigure(gs[1, 1], facecolor='none')
ax = subfig.add_subplot()
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of region trims in OG')
ax.set_ylabel('Number of OGs')
subfig.suptitle('D', x=0.025, y=0.975, fontweight='bold')

# Distribution of number of removed symbols in segment trims (truncated)
idx = int(np.ceil(len(df1) * 0.95))
counts = df1['sym_count'].sort_values(ignore_index=True)[:idx].value_counts()
subfig = fig.add_subfigure(gs[2, 0], facecolor='none')
ax = subfig.add_subplot()
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of non-gap symbols in segment trim')
ax.set_ylabel('Number of\nsegment trims')
subfig.suptitle('E', x=0.025, y=0.975, fontweight='bold')

# Distribution of lengths of region trims
subfig = fig.add_subfigure(gs[2, 1], facecolor='none')
ax = subfig.add_subplot()
ax.hist(df2['length'], bins=100)
ax.set_xlabel('Length of region trim')
ax.set_ylabel('Number of region trims')
subfig.suptitle('F', x=0.025, y=0.975, fontweight='bold')

# Distribution of number of aligned symbols per non-gap symbols in segment trims
subfig = fig.add_subfigure(gs[3, 0], facecolor='none')
ax = subfig.add_subplot()
ax.hist(df1['align_count'] / df1['sym_count'], bins=100)
ax.set_xlabel('Number of aligned non-gap symbols\nper non-gap symbol in trim')
ax.set_ylabel('Number of\nsegment trims')
subfig.suptitle('G', x=0.025, y=0.975, fontweight='bold')

# Distribution of length ratios of total region trims
subfig = fig.add_subfigure(gs[3, 1], facecolor='none')
ax = subfig.add_subplot()
ax.hist(df2['length_ratio'], bins=50)
ax.set_xlabel('Length ratio of region trim')
ax.set_ylabel('Number of region trims')
subfig.suptitle('H', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/insertion_trim.png', dpi=300)
fig.savefig('out/insertion_trim.tiff', dpi=300)
plt.close()
