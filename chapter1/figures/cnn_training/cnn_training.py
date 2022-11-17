"""Make figure of CNN training."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
import tensorflow as tf
from matplotlib.figure import SubplotParams
from matplotlib.gridspec import GridSpec
from src.draw import plot_msa_data, default_sym2color
from src.utils import read_fasta

alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-']
sym2idx = {sym: i for i, sym in enumerate(alphabet)}
ppid_regex = r'ppid=([A-Za-z0-9_.]+)'
spid_regex = r'spid=([a-z]+)'

tree = skbio.read('../../orthology_inference/analysis/ortho_tree/consensus_LG/out/100R_NI.nwk', 'newick', skbio.TreeNode)
tip_order = {tip.name: i for i, tip in enumerate(tree.tips())}

# Load labels
OGid2ppids = {}
ppid2labels = {}
with open('../../orthology_inference/analysis/ortho_MSA/realign_cnn/labels.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        if line.startswith('#'):
            continue
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        if fields['active'] == 'False':
            continue
        OGid, ppid = fields['OGid'], fields['ppid']
        try:
            OGid2ppids[OGid].add(ppid)
        except KeyError:
            OGid2ppids[OGid] = {ppid}
        start, stop, label = int(fields['start']), int(fields['stop']), int(fields['label'])
        try:
            ppid2labels[ppid].append((start, stop, label))
        except KeyError:
            ppid2labels[ppid] = [(start, stop, label)]

# Create records
records = {}
for OGid, ppids in OGid2ppids.items():
    msa1 = read_fasta(f'../../orthology_inference/analysis/ortho_MSA/get_repseqs/out/{OGid}.afa')

    # Convert alignment to indices and make profile
    ppid2idx, msa2 = {}, []
    for i, (header, seq1) in enumerate(msa1):
        ppid = re.search(ppid_regex, header).group(1)
        ppid2idx[ppid] = i
        seq2 = [sym2idx.get(sym, sym2idx['X']) for sym in seq1]  # All non-standard symbols mapped to X
        msa2.append(seq2)
    msa2 = tf.keras.utils.to_categorical(msa2, len(alphabet))
    profile = msa2.sum(axis=0) / len(msa2)

    # Create labels
    for ppid in ppids:
        labels = np.zeros(msa2.shape[1])
        weights = np.zeros(msa2.shape[1])
        for start, stop, label in ppid2labels[ppid]:
            labels[start:stop] = label
            weights[start:stop] = 1
        records[(OGid, ppid)] = (OGid, ppid, profile, msa2[ppid2idx[ppid]], labels, weights)

# Load model and history
df = pd.read_table('../../orthology_inference/analysis/ortho_MSA/realign_cnn/out/history.tsv')
model = tf.keras.models.load_model('../../orthology_inference/analysis/ortho_MSA/realign_cnn/out/model.h5')
layers = {layer.name: layer for layer in model.layers}

if not os.path.exists('out/'):
    os.mkdir('out/')

fig = plt.figure(figsize=(7.5, 4.5), subplotpars=SubplotParams(0.175, 0.2, 0.825, 0.9, 0, 0))
gs = GridSpec(2, 6)

# Count residue labels
positive, negative = 0, 0
for record in records.values():
    labels, weights = record[4], record[5]
    positive += labels.sum()
    negative += weights.sum() - labels.sum()

values = [negative, positive]
labels = [f'{label}\n{value:,g}' for label, value in zip(['negative', 'positive'], values)]
subfig = fig.add_subfigure(gs[0, :2])
ax = subfig.add_subplot()
ax.pie(values, labels=labels, labeldistance=1.5, textprops={'ha': 'center'})
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# Plot embedding
subfig = fig.add_subfigure(gs[1, :3])
ax = subfig.add_subplot()
weights = layers['embedding1'].get_weights()[0]
ax.scatter(weights[:, 0], weights[:, 1], s=60,
           c=[f'#{default_sym2color[sym]}' for sym in alphabet],
           edgecolors=['black' if sym == '-' else 'none' for sym in alphabet])
for sym, weight in zip(alphabet, weights):
    ax.annotate(sym, xy=weight, va='center', ha='center', size=5, fontweight='bold', fontfamily='monospace')
ax.set_xlabel('Embedding axis 1')
ax.set_ylabel('Embedding axis 2')
subfig.suptitle('D', x=0.025, y=0.975, fontweight='bold')

# Plot weights
subfig = fig.add_subfigure(gs[1, 3:])
nested_gs = GridSpec(4, 2, hspace=1, left=0.1, bottom=0.1, right=0.9, top=0.9)
positions = {'conv1_1_0': nested_gs[0, :],
             'conv1_1_1': nested_gs[1, :],
             'conv1_2_0': nested_gs[2, 0], 'conv2_1_0': nested_gs[2, 1],
             'conv1_2_1': nested_gs[3, 0], 'conv2_1_1': nested_gs[3, 1]}
for layer_name in ['conv1_1', 'conv1_2', 'conv2_1']:
    layer = layers[layer_name]
    weights = layer.get_weights()[0]
    for i in range(weights.shape[2]):
        ax = subfig.add_subplot(positions[f'{layer.name}_{i}'])
        ax.imshow(weights[..., i].transpose())
        ax.set_title(f'{layer.name}_{i}', fontdict={'fontsize': 'medium'})
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
subfig.suptitle('E', x=0.025, y=0.975, fontweight='bold')

# Plot curves
subfig = fig.add_subfigure(gs[0, 2:4])
ax = subfig.add_subplot()
ax.plot(df['loss'], label='train')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')
ax.legend()

subfig = fig.add_subfigure(gs[0, 4:])
ax = subfig.add_subplot()
ax.plot(df['binary_accuracy'], label='accuracy')
ax.plot(df['recall'], label='recall')
ax.plot(df['precision'], label='precision')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')
ax.legend()

fig.savefig('out/cnn_training.png', dpi=300)
fig.savefig('out/cnn_training.tiff', dpi=300)
plt.close()

record_ids = [('010D', 'XP_039479706.2', 465, 945, 50),
              ('010D', 'XP_017002872.2', 465, 945, 50),
              ('23D9', 'XP_026832050.1', 625, 1331, 25),
              ('03E7', 'XP_022223442.2', 698, 1656, 20),
              ('1E46', 'XP_034479221.1', 2407, 2761, 50),
              ('2CDB', 'XP_046866895.1', 1763, 2425, 25)]
for OGid, ppid, start, stop, margin in record_ids:
    start, stop = start - margin, stop + margin
    _, _, profile, seq, labels, weights = records[(OGid, ppid)]
    output = tf.squeeze(model([np.expand_dims(profile, 0), np.expand_dims(seq, 0)]))  # Expand and contract dims

    msa = []
    for header, seq in read_fasta(f'../../orthology_inference/analysis/ortho_MSA/get_repseqs/out/{OGid}.afa'):
        msa_ppid = re.search(ppid_regex, header).group(1)
        msa_spid = re.search(spid_regex, header).group(1)
        msa.append({'ppid': msa_ppid, 'spid': msa_spid, 'seq': seq})
    msa = sorted(msa, key=lambda x: tip_order[x['spid']])

    data = [output[start:stop], labels[start:stop], weights[start:stop]]
    msa_labels = [msa_record['ppid'] if msa_record['ppid'] == ppid else '' for msa_record in msa]
    fig = plot_msa_data([msa_record['seq'][start:stop] for msa_record in msa], data, figsize=(7.5, 3.5),
                        x_start=start,
                        msa_labels=msa_labels, msa_ticklength=1, msa_tickwidth=0.25, msa_tickpad=1.1, msa_labelsize=5,
                        height_ratio=0.5, hspace=0.2, data_max=1.1, data_min=-0.1, data_labels=['output', 'label', 'weight'],
                        msa_legend=True, legend_kwargs={'bbox_to_anchor': (0.905, 0.5), 'loc': 'center left', 'fontsize': 8, 'handletextpad': 0.5, 'markerscale': 1.25, 'handlelength': 1})
    fig.text(0.01, 0.99, 'F', fontsize='large', fontweight='bold',
             horizontalalignment='left', verticalalignment='top')
    plt.subplots_adjust(left=0.075, bottom=0.01, right=0.9, top=0.95)
    plt.savefig(f'out/{OGid}_{ppid}.png', dpi=600)
    plt.close()
