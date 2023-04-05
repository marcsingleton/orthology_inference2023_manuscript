"""Make figure of cluster statistics."""

import os

import matplotlib.pyplot as plt
import pandas as pd

# Load genomes
spids = set()
with open('../../orthology_inference/analysis/ortho_cluster2/config/genomes.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        spids.add(fields['spid'])

# Load sequence data
ppid2data = {}
with open('../../orthology_inference/analysis/ortho_search/sequence_data/out/sequence_data.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        ppid2data[fields['ppid']] = (fields['gnid'], fields['spid'])

# Load OGs
rows = []
with open('../../orthology_inference/analysis/ortho_cluster2/add_paralogs/out/clusters.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        component_id, OGid = fields['component_id'], fields['OGid']
        ppids = {node for edge in fields['edges'].split(',') for node in edge.split(':')}
        for ppid in ppids:
            gnid, spid = ppid2data[ppid]
            rows.append({'component_id': component_id, 'OGid': OGid, 'ppid': ppid, 'gnid': gnid, 'spid': spid})
OGs = pd.DataFrame(rows)

groups = OGs.groupby('OGid')
OGidnum = OGs['OGid'].nunique()
ppidnum = OGs['ppid'].nunique()
gnidnum = OGs['gnid'].nunique()

if not os.path.exists('out/'):
    os.mkdir('out/')

fig = plt.figure(figsize=(7.5, 4.5))
gs = plt.GridSpec(2, 2)
rect = (0.25, 0.2, 0.725, 0.7)

counts = OGs[['spid', 'OGid']].drop_duplicates()['spid'].value_counts().sort_index()
xs = list(range(len(counts)))
subfig = fig.add_subfigure(gs[0, 0], facecolor='none')
ax = subfig.add_axes(rect)
ax.set_xmargin(0.025)
ax.bar(xs, counts.values)
ax.set_xticks(xs, counts.index, rotation=60, fontsize=5.5)
ax.set_xlabel('Species')
ax.set_ylabel('Number of associated OGs')
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

counts = groups['spid'].nunique().value_counts()
subfig = fig.add_subfigure(gs[0, 1], facecolor='none')
ax = subfig.add_axes(rect)
ax.bar(counts.index, counts.values)
ax.set_xlabel('Number of species in OG')
ax.set_ylabel('Number of OGs')
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

counts = OGs.groupby('ppid')['OGid'].nunique().value_counts()
subfig = fig.add_subfigure(gs[1, 0], facecolor='none')
ax = subfig.add_axes(rect)
ax.bar(counts.index, counts.values, width=1)
ax.set_xlabel('Number of OGs associated with protein')
ax.set_ylabel('Number of proteins')
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')

counts_OGid = OGs[['spid', 'OGid']].drop_duplicates()['spid'].value_counts().sort_index()
counts_ppid = OGs.groupby('spid')['ppid'].nunique().sort_index()
subfig = fig.add_subfigure(gs[1, 1], facecolor='none')
ax = subfig.add_axes(rect)
ax.scatter(counts_OGid, counts_ppid, s=10)
ax.set_xlabel('Number of associated OGs')
ax.set_ylabel('Number of proteins')
subfig.suptitle('D', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/clusters.png', dpi=300)
fig.savefig('out/clusters.tiff', dpi=300)
plt.close()
