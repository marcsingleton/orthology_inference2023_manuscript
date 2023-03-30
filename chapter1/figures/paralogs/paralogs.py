"""Make figure of paralog statistics."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import SubplotParams

# Load OGs
OGs1 = {}
with open('../../orthology_inference/analysis/ortho_cluster2/cluster4+_graph/out/4clique/clusters.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        ppids = {node for edge in fields['edges'].split(',') for node in edge.split(':')}
        OGs1[fields['OGid']] = ppids

OGs2 = {}
with open('../../orthology_inference/analysis/ortho_cluster2/add_paralogs/out/clusters.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        ppids = {node for edge in fields['edges'].split(',') for node in edge.split(':')}
        OGs2[fields['OGid']] = ppids

# Calculate stats
rows = []
for OGid, OG1 in OGs1.items():
    OG2 = OGs2[OGid]
    rows.append({'OGid': OGid, 'ppidnum1': len(OG1), 'ppidnum2': len(OG2), 'delta': len(OG2 - OG1)})
df = pd.DataFrame(rows)

# Make plots
if not os.path.exists('out/'):
    os.mkdir('out/')

fig = plt.figure(figsize=(7.5, 4.5), subplotpars=SubplotParams(0.15, 0.275, 0.85, 0.9, 0, 0))
gs = plt.GridSpec(2, 2)

counts = (df['delta'] == 0).value_counts()
labels = [('w/o in-paralogs' if idx else 'w/ in-paralogs') + f'\n({value:,})' for idx, value in zip(counts.index, counts.values)]
subfig = fig.add_subfigure(gs[0, 0])
ax = subfig.add_subplot()
ax.pie(counts.values, labels=labels, labeldistance=2, textprops={'ha': 'center'})
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

subfig = fig.add_subfigure(gs[1, 0])
ax = subfig.add_subplot()
hb = ax.hexbin(df['ppidnum1'], df['ppidnum2'], bins='log', gridsize=25, mincnt=1, linewidth=0)
ax.set_xlabel('Number of proteins\nin OG w/o in-paralogs')
ax.set_ylabel('Number of proteins\nin OG w/ in-paralogs')
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')
fig.colorbar(hb, ax=ax)
ax.set_aspect(1)

counts = df.loc[df['delta'] != 0, 'delta'].value_counts()
subfig = fig.add_subfigure(gs[0, 1])
ax = subfig.add_subplot()
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of in-paralogs in OG')
ax.set_ylabel('Number of OGs')
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

subfig = fig.add_subfigure(gs[1, 1])
ax = subfig.add_subplot()
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of in-paralogs in OG')
ax.set_ylabel('Number of OGs')
subfig.suptitle('D', x=0.025, y=0.975, fontweight='bold')
ax.set_yscale('log')

fig.savefig('out/paralogs.png', dpi=300)
fig.savefig('out/paralogs.tiff', dpi=300)
plt.close()
