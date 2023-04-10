"""Make figure of cluster heatmap."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from src2.brownian.linkage import make_tree
from src2.draw import plot_tree

pdidx = pd.IndexSlice
min_length = 30

min_indel_columns = 5  # Indel rates below this value are set to 0
min_aa_rate = 0.5
min_indel_rate = 0.1

# Load regions
rows = []
with open(f'../../IDR_evolution/analysis/IDRpred/region_filter/out/regions_{min_length}.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, disorder = fields['OGid'], int(fields['start']), int(fields['stop']), fields[
            'disorder'] == 'True'
        rows.append({'OGid': OGid, 'start': start, 'stop': stop, 'disorder': disorder})
all_regions = pd.DataFrame(rows)

asr_rates = pd.read_table(f'../../IDR_evolution/analysis/evofit/asr_stats/out/regions_{min_length}/rates.tsv')
asr_rates = all_regions.merge(asr_rates, how='right', on=['OGid', 'start', 'stop'])
row_idx = (asr_rates['indel_num_columns'] < min_indel_columns) | asr_rates['indel_rate_mean'].isna()
asr_rates.loc[row_idx, 'indel_rate_mean'] = 0

row_idx = (asr_rates['aa_rate_mean'] > min_aa_rate) | (asr_rates['indel_rate_mean'] > min_indel_rate)
column_idx = ['OGid', 'start', 'stop', 'disorder']
region_keys = asr_rates.loc[row_idx, column_idx]

models = pd.read_table(f'../../IDR_evolution/analysis/brownian/get_models/out/models_{min_length}.tsv', header=[0, 1])
df = region_keys.merge(models.droplevel(1, axis=1), how='left', on=['OGid', 'start', 'stop'])
df = df.set_index(['OGid', 'start', 'stop', 'disorder'])

feature_groups = {}
feature_labels = []
nonmotif_labels = []
for column_label, group_label in models.columns:
    if not column_label.endswith('_loglikelihood_BM') or group_label == 'ids_group':
        continue
    feature_label = column_label.removesuffix('_loglikelihood_BM')
    try:
        feature_groups[group_label].append(feature_label)
    except KeyError:
        feature_groups[group_label] = [feature_label]
    feature_labels.append(feature_label)
    if group_label != 'motifs_group':
        nonmotif_labels.append(feature_label)

columns = {}
for feature_label in feature_labels:
    columns[f'{feature_label}_AIC_BM'] = 2 * (2 - df[f'{feature_label}_loglikelihood_BM'])
    columns[f'{feature_label}_AIC_OU'] = 2 * (3 - df[f'{feature_label}_loglikelihood_OU'])
    columns[f'{feature_label}_delta_AIC'] = columns[f'{feature_label}_AIC_BM'] - columns[f'{feature_label}_AIC_OU']
    columns[f'{feature_label}_sigma2_ratio'] = df[f'{feature_label}_sigma2_BM'] / df[f'{feature_label}_sigma2_OU']
df = pd.concat([df, pd.DataFrame(columns)], axis=1)

if not os.path.exists('out/'):
    os.mkdir('out/')

# Hierarchical heatmap
legend_args = {'aa_group': ('Amino acid content', 'grey', ''),
               'charge_group': ('Charge properties', 'black', ''),
               'physchem_group': ('Physiochemical properties', 'white', ''),
               'complexity_group': ('Repeats and complexity', 'white', 4 * '.'),
               'motifs_group': ('Motifs', 'white', 4 * '\\')}
group_labels = ['aa_group', 'charge_group', 'motifs_group', 'physchem_group', 'complexity_group']
group_labels_nonmotif = ['aa_group', 'charge_group', 'physchem_group', 'complexity_group']
gridspec_kw = {'width_ratios': [0.1, 0.9], 'wspace': 0,
               'height_ratios': [0.975, 0.025], 'hspace': 0.01,
               'left': 0.05, 'right': 0.95, 'top': 0.95, 'bottom': 0.125}

column_labels = []
for group_label in group_labels:
    column_labels.extend([f'{feature_label}_delta_AIC' for feature_label in feature_groups[group_label]])
array = np.nan_to_num(df.loc[pdidx[:, :, :, True], column_labels].to_numpy(), nan=1)  # Re-arrange and convert to array

cdm = pdist(array, metric='correlation')
lm = linkage(cdm, method='average')

# Convert to tree and get branch colors
tree = make_tree(lm)
tip_order = [tip.name for tip in tree.tips()]
node2color, node2tips = {}, {}
for node in tree.postorder():
    if node.is_tip():
        tips = 1
    else:
        tips = sum([node2tips[child] for child in node.children])
    node2tips[node] = tips
    node2color[node] = str(max(0, (11 - tips) / 10))

fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5), gridspec_kw=gridspec_kw)

# Tree
ax = axs[0, 0]
plot_tree(tree, ax=ax, linecolor=node2color, linewidth=0.2, tip_labels=False,
          xmin_pad=0.025, xmax_pad=0, ymin_pad=1 / (2 * len(array)), ymax_pad=1 / (2 * len(array)))
ax.set_ylabel('Disorder regions')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Heatmap
ax = axs[0, 1]
im = ax.imshow(array[tip_order], aspect='auto', cmap=plt.colormaps['inferno'], interpolation='none')
ax.xaxis.set_label_position('top')
ax.set_xlabel('Features')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Corner axis
ax = axs[1, 0]
ax.set_visible(False)

# Legend
ax = axs[1, 1]
x = 0
handles = []
for group_label in group_labels:
    label, color, hatch = legend_args[group_label]
    dx = len(feature_groups[group_label]) / len(column_labels)
    rectangle = mpatches.Rectangle((x, 0), dx, 1, label=label, facecolor=color, hatch=hatch,
                                   edgecolor='black', linewidth=0.75, clip_on=False)
    ax.add_patch(rectangle)
    handles.append(rectangle)
    x += dx
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.25, 0), fontsize=8)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Colorbar
xcenter = 0.75
width = 0.2
ycenter = gridspec_kw['bottom'] / 2
height = 0.015
cax = fig.add_axes((xcenter - width / 2, ycenter - height / 2, width, height))
cax.set_title('$\mathregular{AIC_{BM} - AIC_{OU}}$', fontdict={'fontsize': 10})
fig.colorbar(im, cax=cax, orientation='horizontal')

fig.savefig(f'out/hierarchy.png', dpi=600)
plt.close()
