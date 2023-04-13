"""Make figure for Brownian motion analyses of features."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
from matplotlib.lines import Line2D
from matplotlib.patches import ArrowStyle
from matplotlib.transforms import blended_transform_factory
from sklearn.decomposition import PCA
from src2.brownian.pca import add_pca_arrows
from src2.draw import plot_msa
from src2.utils import read_fasta


def zscore(df):
    return (df - df.mean()) / df.std()


def get_bin2data(hb, xs, ys):
    offsets = hb.get_offsets()
    xys = np.stack([xs, ys]).transpose()
    dist2 = ((offsets[:, np.newaxis, :] - xys) ** 2).sum(axis=-1)
    bin_idxs = np.argmin(dist2, axis=0)

    bin2data = {}
    for data_idx, bin_idx in enumerate(bin_idxs):
        try:
            bin2data[bin_idx].append(data_idx)
        except KeyError:
            bin2data[bin_idx] = [data_idx]

    return bin2data


def get_bin(hb, x, y):
    bin2data = get_bin2data(hb, [x], [y])
    return list(bin2data)[0]


def set_bin_edge(ax, hb, bin_idx, facecolor, edgecolor, linewidth):
    path = hb.get_paths()[0]
    offset = hb.get_offsets()[bin_idx]

    vertices = path.vertices + offset
    hexagon = plt.Polygon(vertices, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(hexagon)


pdidx = pd.IndexSlice
spid_regex = r'spid=([a-z]+)'
min_length = 30

pca_components = 10
cmap = plt.colormaps['Greys']
hexbin_kwargs_log = {'gridsize': 30, 'mincnt': 1, 'linewidth': 0, 'bins': 'log'}
handle_markerfacecolor = 0.6
legend_kwargs = {'fontsize': 6.5, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
arrow_colors = ['#4e79a7', '#f28e2b', '#e15759', '#499894', '#59a14f', '#f1ce63', '#b07aa1', '#d37295',
                '#a0cbe8', '#ffbe7d', '#ff9d9a', '#86bcb6', '#8cd17d', '#b6992d', '#d4a6c8', '#fabfd2']
arrow_scale = 0.99
arrowstyle_kwargs = ArrowStyle('simple', head_length=8, head_width=8, tail_width=2.5)
dpi = 400

tree_template = skbio.read('../../IDR_evolution/data/trees/consensus_LG/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree_template.tips())}

# Load features
all_features = pd.read_table('../../IDR_evolution/analysis/brownian/get_features/out/features.tsv', header=[0, 1])
all_features.loc[all_features[('kappa', 'charge_group')] == -1, 'kappa'] = 1  # Need to specify full column index to get slicing to work
all_features.loc[all_features[('omega', 'charge_group')] == -1, 'omega'] = 1
all_features['length'] = all_features['length'] ** 0.6
all_features.rename(columns={'length': 'radius_gyration'}, inplace=True)

feature_labels = [feature_label for feature_label, group_label in all_features.columns
                  if group_label != 'ids_group']
nonmotif_labels = [feature_label for feature_label, group_label in all_features.columns
                   if group_label not in ['ids_group', 'motifs_group']]
all_features = all_features.droplevel(1, axis=1)

# Load regions as segments
rows = []
with open(f'../../IDR_evolution/analysis/IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields['disorder'] == 'True'
        for ppid in fields['ppids'].split(','):
            rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder, 'ppid': ppid})
all_segments = pd.DataFrame(rows)
all_regions = all_segments.drop('ppid', axis=1).drop_duplicates()

# Load and format data
asr_rates = pd.read_table(f'../../IDR_evolution/analysis/evofit/asr_stats/out/regions_{min_length}/rates.tsv')
asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])

features = all_segments.merge(all_features, how='left', on=['OGid', 'start', 'stop', 'ppid'])
features = features.groupby(['OGid', 'start', 'stop', 'disorder']).mean()
features = all_regions.merge(features, how='left', on=['OGid', 'start', 'stop', 'disorder'])
features = features.set_index(['OGid', 'start', 'stop', 'disorder'])

roots = pd.read_table(f'../../IDR_evolution/analysis/brownian/get_contrasts/out/features/roots_{min_length}.tsv', skiprows=[1])  # Skip group row
roots = all_regions.merge(roots, how='left', on=['OGid', 'start', 'stop'])
roots = roots.set_index(['OGid', 'start', 'stop', 'disorder'])

contrasts = pd.read_table(f'../../IDR_evolution/analysis/brownian/get_contrasts/out/features/contrasts_{min_length}.tsv', skiprows=[1])  # Skip group row
contrasts = all_regions.merge(contrasts, how='left', on=['OGid', 'start', 'stop'])
contrasts = contrasts.set_index(['OGid', 'start', 'stop', 'disorder', 'contrast_id'])

rates = (contrasts ** 2).groupby(['OGid', 'start', 'stop', 'disorder']).mean()

if not os.path.exists(f'out/'):
    os.mkdir(f'out/')

# === MAIN FIGURE ROOTS ===
fig_width = 7.5
fig_height = 3
gs = plt.GridSpec(1, 2)
rectA = (0.15, 0.15, 0.55, 0.75)
rectB = (0.15, 0.15, 0.55, 0.75)

roots_nonmotif = roots[nonmotif_labels]
disorder = roots.loc[pdidx[:, :, :, :], :]
disorder_nonmotif = disorder[nonmotif_labels]

data = zscore(disorder_nonmotif)
pca = PCA(n_components=pca_components)
transform = pca.fit_transform(data.to_numpy())

fig = plt.figure(figsize=(fig_width, fig_height))

# --- PANEL A ---
subfig = fig.add_subfigure(gs[0, 0])
ax = subfig.add_axes(rectA)
hb = ax.hexbin(transform[:, 0], transform[:, 1], cmap=cmap, **hexbin_kwargs_log)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
handles = [Line2D([], [], label='disorder', marker='h', markerfacecolor=cmap(handle_markerfacecolor),
                  markersize=8, markeredgecolor='none', linestyle='none')]
ax.legend(handles=handles)
subfig.colorbar(hb, cax=ax.inset_axes((1.1, 0, 0.05, 1)))
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# --- PANEL B ---
subfig = fig.add_subfigure(gs[0, 1])
ax = subfig.add_axes(rectB)
ax.hexbin(transform[:, 0], transform[:, 1], cmap=cmap, **hexbin_kwargs_log)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
add_pca_arrows(ax, pca, data.columns, 0, 1,
               legend_kwargs=legend_kwargs,
               arrow_scale=arrow_scale, arrow_colors=arrow_colors, arrowstyle_kwargs=arrowstyle_kwargs)
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

fig.savefig('out/root.png', dpi=dpi)
fig.savefig('out/root.tiff', dpi=dpi)

# === MAIN FIGURE RATES ===
fig_width = 7.5
fig_height = 8
gs = plt.GridSpec(4, 2, height_ratios=[1.5, 1.5, 1, 1])
rectA = (0.15, 0.2, 0.55, 0.75)
rectB = (0.175, 0.2, 0.55, 0.75)

plot_msa_kwargs_common = {'tree_kwargs': {'linewidth': 0.75, 'tip_labels': False, 'xmin_pad': 0.025, 'xmax_pad': 0.025},
                          'msa_legend': True}
plot_msa_kwargs1 = {'left': 0.17, 'right': 0.85, 'top': 0.95, 'bottom': 0.025, 'anchor': (0, 0.5),
                    'hspace': 0.01,
                    'tree_position': 0.02, 'tree_width': 0.15,
                    'legend_kwargs': {'bbox_to_anchor': (0.875, 0.5), 'loc': 'center left', 'fontsize': 5,
                                      'handletextpad': 0.5, 'markerscale': 0.75, 'handlelength': 1},
                    **plot_msa_kwargs_common}
plot_msa_kwargs2 = {'left': 0.16, 'right': 0.935, 'top': 0.875, 'bottom': 0.125, 'anchor': (0, 0.5),
                    'hspace': 0.01,
                    'tree_position': 0.01, 'tree_width': 0.15,
                    'legend_kwargs': {'bbox_to_anchor': (0.9375, 0.5), 'loc': 'center left', 'fontsize': 5,
                                      'handletextpad': 0.5, 'markerscale': 0.75, 'handlelength': 1},
                    **plot_msa_kwargs_common}
records = {('0A8A', 1102, 1200): {'panel_label': 'E', 'fig_width_ratio': 0.5,
                                  'color': 'C0',
                                  'gs': gs[2, 0], 'plot_msa_kwargs': plot_msa_kwargs1},
           ('3139', 249, 339):   {'panel_label': 'F', 'fig_width_ratio': 0.5,
                                  'color': 'C2',
                                  'gs': gs[2, 1], 'plot_msa_kwargs': plot_msa_kwargs1},
           ('04B0', 0, 159):     {'panel_label': 'G', 'fig_width_ratio': 1,
                                  'color': 'C1',
                                  'gs': gs[3, :], 'plot_msa_kwargs': plot_msa_kwargs2}}

rates_nonmotif = rates[nonmotif_labels]
disorder = rates.loc[pdidx[:, :, :, :], :]
disorder_nonmotif = disorder[nonmotif_labels]

data = zscore(disorder_nonmotif)
pca = PCA(n_components=pca_components)
transform = pca.fit_transform(data.to_numpy())
ids2idx = {ids: idx for idx, ids in enumerate(data.index.droplevel('disorder'))}

fig = plt.figure(figsize=(fig_width, fig_height))

# --- PANEL A ---
subfig = fig.add_subfigure(gs[0, 0])
ax = subfig.add_axes(rectA)
hb = ax.hexbin(transform[:, 0], transform[:, 1], cmap=cmap, **hexbin_kwargs_log)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
handles = [Line2D([], [], label='disorder', marker='h', markerfacecolor=cmap(handle_markerfacecolor),
                  markersize=8, markeredgecolor='none', linestyle='none')]
ax.legend(handles=handles)
subfig.colorbar(hb, cax=ax.inset_axes((1.1, 0, 0.05, 1)))
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# --- PANEL B ---
pca = PCA(n_components=pca_components)
df = data.merge(asr_rates, left_index=True, right_on=['OGid', 'start', 'stop', 'disorder'])
transform = pca.fit_transform(df[data.columns].to_numpy())

xs = transform[:, 0]
ys = df['aa_rate_mean'] + df['indel_rate_mean']

subfig = fig.add_subfigure(gs[0, 1])
ax = subfig.add_axes(rectB)
hb = ax.hexbin(xs, ys, **hexbin_kwargs_log)
ax.set_xlabel('PC1')
ax.set_ylabel('Total substitution rate')
subfig.colorbar(hb, cax=ax.inset_axes((1.1, 0, 0.05, 1)))
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')

# --- PANEL C ---
subfig = fig.add_subfigure(gs[1, 0])
ax = subfig.add_axes(rectA)
hb = ax.hexbin(transform[:, 1], transform[:, 2], cmap=cmap, **hexbin_kwargs_log)
ax.set_xlabel('PC2')
ax.set_ylabel('PC3')
handles = [Line2D([], [], label='disorder', marker='h', markerfacecolor=cmap(handle_markerfacecolor),
                  markersize=8, markeredgecolor='none', linestyle='none')]
ax.legend(handles=handles)
subfig.colorbar(hb, cax=ax.inset_axes((1.1, 0, 0.05, 1)))
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')

for ids, params in records.items():
    data_idx = ids2idx[ids]
    x, y = transform[data_idx, 1], transform[data_idx, 2]
    bin_idx = get_bin(hb, x, y)
    set_bin_edge(ax, hb, bin_idx, 'none', params['color'], 1.5)

# --- PANEL D ---
subfig = fig.add_subfigure(gs[1, 1])
ax = subfig.add_axes(rectB)
ax.hexbin(transform[:, 1], transform[:, 2], cmap=cmap, **hexbin_kwargs_log)
ax.set_xlabel('PC2')
ax.set_ylabel('PC3')
add_pca_arrows(ax, pca, data.columns, 1, 2,
               legend_kwargs=legend_kwargs,
               arrow_scale=arrow_scale, arrow_colors=arrow_colors, arrowstyle_kwargs=arrowstyle_kwargs)
subfig.suptitle('D', x=0.025, y=0.975, fontweight='bold')

# --- PANELS E - G ---
for (OGid, start, stop), params in records.items():
    # Load MSA
    msa = []
    for header, seq in read_fasta(f'../../IDR_evolution/data/alignments/fastas/{OGid}.afa'):
        spid = re.search(spid_regex, header).group(1)
        msa.append({'spid': spid, 'seq': seq})
    msa = sorted(msa, key=lambda x: tip_order[x['spid']])

    # Load tree and convert to vectors at tips
    tree = tree_template.shear([record['spid'] for record in msa])
    for node in tree.postorder():  # Ensure tree is ordered as in original
        if node.is_tip():
            node.value = tip_order[node.name]
        else:
            node.children = sorted(node.children, key=lambda x: x.value)
            node.value = sum([child.value for child in node.children])

    subfig = fig.add_subfigure(params['gs'])
    plot_msa([record['seq'][start:stop] for record in msa],
             x_start=start,
             fig=subfig, figsize=(fig_width * params['fig_width_ratio'], fig_height / gs.nrows),
             tree=tree,
             **params['plot_msa_kwargs'])
    subfig.suptitle(params['panel_label'], x=0.0125 / params['fig_width_ratio'], y=0.975, fontweight='bold')

    for ax in subfig.axes:
        bar_offset = 0.0075 / params['fig_width_ratio']
        bar_width = 0.005 / params['fig_width_ratio']
        bar_ax = ax.inset_axes((params['plot_msa_kwargs']['tree_position'] - bar_offset, 0, bar_width, 1),
                               transform=blended_transform_factory(subfig.transSubfigure, ax.transAxes))
        bar_ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=params['color']))
        bar_ax.set_axis_off()

fig.savefig('out/rate.png', dpi=dpi)
fig.savefig('out/rate.tiff', dpi=dpi)

"""
Here are some scratch work for ranking candidate regions.

CHARGE
('0A8A', 1102, 1200) **
('289C', 102, 213) **

GLYCINE
('190D', 317, 399) **
('3139', 249, 339) **
('20CF', 695, 751) **
('0DA4', 1591, 1683) *

Q RICH
('1A58', 702, 788) **
('3132', 1234, 1375) **
('301D', 720, 856) *
('04B0', 0, 159) *
('0DE8', 0, 86) *
('2099', 739, 811) *
('210F', 0, 227) .
('001D', 1437, 1515) .
('022B', 795, 997) .
"""