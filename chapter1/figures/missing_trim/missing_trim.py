"""Make figure of missing HMM trimming data."""

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

df = pd.read_table('../../orthology_inference/analysis/ortho_MSA/missing_trim/out/trim_stats.tsv')

df['length'] = df['stop'] - df['start']
df['length_ratio'] = df['length'] / df['colnum']
ppid_groups = df.groupby(['OGid', 'ppid'])  # Use combined key in case ppid in multiple OGids
OGid_groups = df.groupby(['OGid'])

if not os.path.exists('out/'):
    os.mkdir('out/')

fig = plt.figure(figsize=(7.5, 4), subplotpars=SubplotParams(0.25, 0.325, 0.9, 0.925))
gs = GridSpec(2, 2)

# Number of OGs with trims
values = [len(set(OGids)) - df['OGid'].nunique(), df['OGid'].nunique()]
labels = [f'{label}\n({value:,})' for label, value in zip(['no missing trims', 'missing trims'], values)]
subfig = fig.add_subfigure(gs[0, 0], facecolor='none')
ax = subfig.add_axes((0.1, 0.1, 0.8, 0.8))
ax.pie(values, labels=labels, labeldistance=1.75, textprops={'ha': 'center'})
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# Distribution of number sequences with trims in OGs
counts = OGid_groups['ppid'].nunique().value_counts()
subfig = fig.add_subfigure(gs[0, 1], facecolor='none')
ax = subfig.add_subplot()
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of unique sequences with trims in OG')
ax.set_ylabel('Number of OGs')
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

# Distribution of length of trims
subfig = fig.add_subfigure(gs[1, 0], facecolor='none')
ax = subfig.add_subplot()
ax.hist(df['length'], bins=100)
ax.set_xlabel('Length of trim')
ax.set_ylabel('Number of trims')
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')

# Distribution of length ratio of trims
subfig = fig.add_subfigure(gs[1, 1], facecolor='none')
ax = subfig.add_subplot()
ax.hist(df['length_ratio'], bins=50)
ax.set_xlabel('Length ratio of trim')
ax.set_ylabel('Number of trims')
subfig.suptitle('D', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/missing_trim.png', dpi=300)
fig.savefig('out/missing_trim.tiff', dpi=300)
plt.close()
