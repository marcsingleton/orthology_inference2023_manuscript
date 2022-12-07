"""Make figure of HMM training."""

import os
import json

import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

spid_regex = r'spid=([a-z]+)'
state_labels = ['1A', '1B', '2', '3']
state_colors = ['C0', 'C3', 'C1', 'C2']

# Load labels
OGid2labels = {}
label_set = set()
with open('../../orthology_inference/analysis/ortho_MSA/insertion_hmm/labels.tsv') as file:
    field_names = file.readline().rstrip('\n').split('\t')
    for line in file:
        fields = {key: value for key, value in zip(field_names, line.rstrip('\n').split('\t'))}
        OGid, start, stop, label = fields['OGid'], int(fields['start']), int(fields['stop']), fields['label']
        label_set.add(label)
        try:
            OGid2labels[OGid].append((start, stop, label))
        except KeyError:
            OGid2labels[OGid] = [(start, stop, label)]

if set(state_labels) != label_set:
    raise RuntimeError('label_set is not equal to set of state_labels')

# Load history and model parameters
with open('../../orthology_inference/analysis/ortho_MSA/insertion_hmm/out/history.json') as file:
    history = json.load(file)
with open('../../orthology_inference/analysis/ortho_MSA/insertion_hmm/out/model.json') as file:
    model_json = json.load(file)

if not os.path.exists('out/'):
    os.mkdir('out/')

fig = plt.figure(figsize=(7.5, 5.5), subplotpars=SubplotParams(0.3, 0.225, 0.9, 0.9, 0, 0))
gs = GridSpec(2, 4, height_ratios=[1, 1.5])

# Plot state distribution
counts = {label: 0 for label in state_labels}
for labels in OGid2labels.values():
    for start, stop, label in labels:
        counts[label] += stop - start
values = [counts[label] for label in state_labels]
labels = [f'{label}\n{value:,}' for label, value in zip(state_labels, values)]
subfig = fig.add_subfigure(gs[0, :2])
ax = subfig.add_subplot()
ax.pie(values, colors=state_colors, labels=labels, labeldistance=1.3, textprops={'ha': 'center'})
subfig.suptitle('A', x=0.025, y=0.975, fontweight='bold')

# Plot loss curve
xs = [record['iter_num'] for record in history]
ys = [record['ll'] for record in history]
subfig = fig.add_subfigure(gs[0, 2:])
ax = subfig.add_subplot()
ax.plot(xs, ys)
ax.set_xlabel('Iteration')
ax.set_ylabel('Conditional\nlog-likelihood')
subfig.suptitle('B', x=0.025, y=0.975, fontweight='bold')

# Plot model parameters
params = ['pi', 'q0', 'q1']
subfig = fig.add_subfigure(gs[1, 0], facecolor='none')
axs = subfig.subplots(len(params), 1, sharex=True, gridspec_kw={'hspace': 0.15})
for label, color in zip(state_labels, state_colors):
    for ax, param in zip(axs, params):
        xs = [record['iter_num'] for record in history]
        ys = [record['e_dists_norm'][label][param] for record in history]
        ax.plot(xs, ys, label=label, color=color)
        ax.set_ylabel(param)
axs[-1].set_xlabel('Iteration')
subfig.suptitle('C', x=0.025, y=0.975, fontweight='bold')

params = ['p0', 'p1']
subfig = fig.add_subfigure(gs[1, 1], facecolor='none')
axs = subfig.subplots(len(params), 1, sharex=True, gridspec_kw={'hspace': 0.15})
for label, color in zip(state_labels, state_colors):
    for ax, param in zip(axs, params):
        xs = [record['iter_num'] for record in history]
        ys = [record['e_dists_norm'][label][param] for record in history]
        ax.plot(xs, ys, label=label, color=color)
        ax.set_ylabel(param)
axs[-1].set_xlabel('Iteration')
subfig.suptitle('D', x=0.025, y=0.975, fontweight='bold')

params = ['a', 'b']
subfig = fig.add_subfigure(gs[1, 2], facecolor='none')
axs = subfig.subplots(len(params), 1, sharex=True, gridspec_kw={'hspace': 0.15})
for label, color in zip(state_labels, state_colors):
    for ax, param in zip(axs, params):
        xs = [record['iter_num'] for record in history]
        ys = [record['e_dists_norm'][label][param] for record in history]
        ax.plot(xs, ys, label=label, color=color)
        ax.set_ylabel(param)
axs[-1].set_xlabel('Iteration')
subfig.suptitle('E', x=0.025, y=0.975, fontweight='bold')

subfig = fig.add_subfigure(gs[1, 3], facecolor='none')
axs = subfig.subplots(len(state_labels), 1, sharex=True, gridspec_kw={'hspace': 0.15})
for label, color in zip(state_labels, state_colors):
    for ax, param in zip(axs, state_labels):
        if param == label:
            continue
        xs = [record['iter_num'] for record in history]
        ys = [record['t_dists_norm'][label][param] for record in history]
        ax.plot(xs, ys, label=label, color=color)
        ax.set_ylabel(param)
for ax in axs:
    ax.set_yscale('log')
axs[-1].set_xlabel('Iteration')
subfig.suptitle('F', x=0.025, y=0.975, fontweight='bold')

handles = [Line2D([], [], label=label, color=color) for label, color in zip(state_labels, state_colors)]
fig.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.5, -0.01), loc='lower center')

fig.savefig('out/hmm_training.png', dpi=300)
fig.savefig('out/hmm_training.tiff', dpi=300)
plt.close()
