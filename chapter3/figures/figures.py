"""Make all figures from training tutorial."""

# In[1]:
import os
import pprint
import random
from functools import reduce
from itertools import accumulate

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from homomorph import HMM
from numpy import exp, log
from sklearn.metrics import roc_curve
from utils import fit_CML

dpi = 300
legend_kwargs = {'frameon': False, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}

if not os.path.exists('out/'):
    os.mkdir('out/')


# In[2]:
t_dists = {1: {1: 0.95, 2: 0.05},
           2: {1: 0.05, 2: 0.9, 3: 0.05},
           3: {2: 0.35, 3: 0.65}}
e_dists = {1: {'A': 1},
           2: {'A': 0.5, 'B': 0.5},
           3: {'B': 1}}
start_dist = {1: 0.2, 2: 0.5, 3: 0.3}

model = HMM(t_dists=t_dists, e_dists=e_dists, start_dist=start_dist)


# In[3]:
data = [model.simulate(200, random_state=i) for i in range(10)]  # Use a different random seed for each example
data = [list(zip(*example)) for example in data]


# In[4]:
# Make transition count dicts and add pseudocounts
t_pseudo = 0.1
t_counts = {}
for state1, t_dist in t_dists.items():
    t_count = {}
    for state2 in t_dist:
        t_count[state2] = t_pseudo
    t_counts[state1] = t_count

# Add observed counts
for example in data:
    xs, ys = example
    state0 = xs[0]
    for state1 in xs[1:]:
        t_counts[state0][state1] += 1
        state0 = state1

# Normalize counts
t_dists_hat = {}
for state1, t_count in t_counts.items():
    t_sum = sum(t_count.values())
    t_dist_hat = {}
    for state2, count in t_count.items():
        t_dist_hat[state2] = count / t_sum
    t_dists_hat[state1] = t_dist_hat


# In[5]:
# Collect all possible emissions
e_set = set()
for example in data:
    xs, ys = example
    e_set.update(ys)

# Make emission count dicts and add pseudocounts
e_pseudo = 0.1
e_counts = {}
for state in t_dists:
    e_counts[state] = {emit: e_pseudo for emit in e_set}

# Add observed counts
for example in data:
    xs, ys = example
    for state, emit in zip(xs, ys):
        e_counts[state][emit] += 1

# Normalize counts
e_dists_hat = {}
for state, e_count in e_counts.items():
    e_sum = sum(e_count.values())
    e_dist_hat = {}
    for emit, count in e_count.items():
        e_dist_hat[emit] = count / e_sum
    e_dists_hat[state] = e_dist_hat


# In[6]:
# Make start count dicts and add pseudocounts
start_pseudo = 0.1
start_count = {}
for state in start_dist:
    start_count[state] = start_pseudo

# Add observed counts
for example in data:
    xs, ys = example
    start_count[xs[0]] += 1

# Normalize counts
start_sum = sum(start_count.values())
start_dist_hat = {}
for state, count in start_count.items():
    start_dist_hat[state] = count / start_sum


# In[7]:
model_hat = HMM(t_dists=t_dists_hat, e_dists=e_dists_hat, start_dist=start_dist_hat)


# In[8]:
xs, ys = data[0]
fbs = model.forward_backward(ys)
fbs_hat = model_hat.forward_backward(ys)

fig, axs = plt.subplots(4, 1, figsize=(6.4, 6), sharex=True, layout='constrained')

axs[0].plot(ys)
for state in t_dists:
    axs[1].plot([x == state for x in xs], label=state)
for state, line in sorted(fbs.items()):
    axs[2].plot(line, label=state)
for state, line in sorted(fbs_hat.items()):
    axs[3].plot(line, label=state)
axs[3].set_xlabel('Time step')
axs[0].set_ylabel('Emission')
axs[1].set_ylabel('Label')
axs[2].set_ylabel('Probability')
axs[3].set_ylabel('Probability')
axs[1].legend(title='true states', **legend_kwargs)
axs[2].legend(title='model', **legend_kwargs)
axs[3].legend(title='model_hat', **legend_kwargs)

fig.savefig('out/categorical.png', dpi=dpi)
fig.savefig('out/categorical.tiff', dpi=dpi)
plt.close()


# In[9]:
t_dists = {1: {1: 0.95, 2: 0.05},
           2: {1: 0.25, 2: 0.75}}
e_dists = {1: stats.poisson(3),
           2: stats.poisson(0.5)}
start_dist = {1: 0.5, 2: 0.5}

model = HMM(t_dists=t_dists, e_dists=e_dists, start_dist=start_dist)

data = [model.simulate(200, random_state=i) for i in range(10)]  # Use a different random seed for each example
data = [list(zip(*example)) for example in data]


# In[10]:
# Make emission dicts keyed by state
state2emits = {}
for state in t_dists:
    state2emits[state] = []

# Add emissions
for example in data:
    xs, ys = example
    for state, emit in zip(xs, ys):
        state2emits[state].append(emit)

# Average emissions
lambda_hats = {}
for state, emits in state2emits.items():
    lambda_hat = sum(emits) / len(emits)
    lambda_hats[state] = lambda_hat


# In[11]:
e_dists_hat = {state: stats.poisson(lambda_hat) for state, lambda_hat in lambda_hats.items()}


# In[12]:
# Make transition count dicts and add pseudocounts
t_pseudo = 0.1
t_counts = {}
for state1, t_dist in t_dists.items():
    t_count = {}
    for state2 in t_dist:
        t_count[state2] = t_pseudo
    t_counts[state1] = t_count

# Add observed counts
for example in data:
    xs, ys = example
    state0 = xs[0]
    for state1 in xs[1:]:
        t_counts[state0][state1] += 1
        state0 = state1

# Normalize counts
t_dists_hat = {}
for state1, t_count in t_counts.items():
    t_sum = sum(t_count.values())
    t_dist_hat = {}
    for state2, count in t_count.items():
        t_dist_hat[state2] = count / t_sum
    t_dists_hat[state1] = t_dist_hat


# In[13]:
# Make start count dicts and add pseudocounts
start_pseudo = 0.1
start_count = {}
for state in start_dist:
    start_count[state] = start_pseudo

# Add observed counts
for example in data:
    xs, ys = example
    start_count[xs[0]] += 1

# Normalize counts
start_sum = sum(start_count.values())
start_dist_hat = {}
for state, count in start_count.items():
    start_dist_hat[state] = count / start_sum


# In[14]:
model_hat = HMM(t_dists=t_dists_hat, e_dists=e_dists_hat, start_dist=start_dist_hat)

xs, ys = data[0]
fbs = model.forward_backward(ys)
fbs_hat = model_hat.forward_backward(ys)

fig, axs = plt.subplots(4, 1, figsize=(6.4, 6), sharex=True, layout='constrained')

axs[0].plot(ys)
for state in t_dists:
    axs[1].plot([x == state for x in xs], label=state)
for state, line in sorted(fbs.items()):
    axs[2].plot(line, label=state)
for state, line in sorted(fbs_hat.items()):
    axs[3].plot(line, label=state)
axs[3].set_xlabel('Time step')
axs[0].set_ylabel('Emission')
axs[1].set_ylabel('Label')
axs[2].set_ylabel('Probability')
axs[3].set_ylabel('Probability')
axs[1].legend(title='true states', **legend_kwargs)
axs[2].legend(title='model', **legend_kwargs)
axs[3].legend(title='model_hat', **legend_kwargs)

fig.savefig('out/poisson.png', dpi=dpi)
fig.savefig('out/poisson.tiff', dpi=dpi)
plt.close()


# In[15]:
t_dists = {1: {1: 0.95, 2: 0.05},
           2: {1: 0.05, 2: 0.9, 3: 0.05},
           3: {2: 0.35, 3: 0.65}}
e_dists = {1: {'A': 1},
           2: {'A': 0.5, 'B': 0.5},
           3: {'B': 1}}
start_dist = {1: 0.2, 2: 0.5, 3: 0.3}

model = HMM(t_dists=t_dists, e_dists=e_dists, start_dist=start_dist)

data = [model.simulate(200, random_state=i) for i in range(10)]  # Use a different random seed for each example
data = [list(zip(*example)) for example in data]


# In[16]:
random.seed(1)

# Make transition count dicts and add pseudocounts
t_counts = {}
for state1, t_dist in t_dists.items():
    t_count = {}
    for state2 in t_dist:
        t_count[state2] = random.random()
    t_counts[state1] = t_count

# Normalize counts
t_dists_hat = {}
for state1, t_count in t_counts.items():
    t_sum = sum(t_count.values())
    t_dist_hat = {}
    for state2, count in t_count.items():
        t_dist_hat[state2] = count / t_sum
    t_dists_hat[state1] = t_dist_hat


# In[17]:
random.seed(2)

# Collect all possible emissions
e_set = set()
for example in data:
    xs, ys = example
    e_set.update(ys)

# Make emission count dicts and add pseudocounts
e_counts = {}
for state in t_dists:
    e_counts[state] = {emit: random.random() for emit in e_set}

# Normalize counts
e_dists_hat = {}
for state, e_count in e_counts.items():
    e_sum = sum(e_count.values())
    e_dist_hat = {}
    for emit, count in e_count.items():
        e_dist_hat[emit] = count / e_sum
    e_dists_hat[state] = e_dist_hat


# In[18]:
random.seed(3)

# Make start count dicts and add pseudocounts
start_count = {}
for state in start_dist:
    start_count[state] = random.random()

# Normalize counts
start_sum = sum(start_count.values())
start_dist_hat = {}
for state, count in start_count.items():
    start_dist_hat[state] = count / start_sum


# In[19]:
epsilon = 0.01
maxiter = 100

ll0 = None
model_hat = HMM(t_dists=t_dists_hat, e_dists=e_dists_hat, start_dist=start_dist_hat)
for numiter in range(maxiter):
    # Initialize count dictionaries
    ps = []
    t_counts = {state1: {state2: 0 for state2 in t_dist} for state1, t_dist in t_dists_hat.items()}
    e_counts = {state: {emit: 0 for emit in e_dist} for state, e_dist in e_dists_hat.items()}
    start_count = {state: 0 for state in start_dist_hat}

    # Get counts across all examples
    for example in data:
        xs, ys = example
        fs, ss_f = model_hat.forward(ys)
        bs, ss_b = model_hat.backward(ys)

        p = reduce(lambda x, y: x + y, map(log, ss_f))
        ss_f = list(accumulate(map(log, ss_f)))
        ss_b = list(accumulate(map(log, ss_b[::-1])))[::-1]
        ps.append(p)

        # t_counts
        for t in range(len(ys) - 1):
            for state1, t_count in t_counts.items():
                for state2 in t_count:
                    term1 = fs[state1][t] * t_dists_hat[state1][state2]
                    term2 = e_dists_hat[state2][ys[t+1]] * bs[state2][t+1]
                    count = term1 * term2
                    t_count[state2] += count * exp(ss_f[t] + ss_b[t+1] - p)

        # e_counts
        for t in range(len(ys)):
            for state, e_count in e_counts.items():
                if ys[t] in e_count:
                    count = fs[state][t] * bs[state][t]
                    e_count[ys[t]] += count * exp(ss_f[t] + ss_b[t] - p)

        # start_count
        for state in start_count:
            count = fs[state][0] * bs[state][0]
            start_count[state] += count * exp(ss_f[0] + ss_b[0] - p)

    # Format parameters for display
    t_string = pprint.pformat(t_dists_hat).replace('\n', '\n' + len('t_dists: ') * ' ')
    e_string = pprint.pformat(e_dists_hat).replace('\n', '\n' + len('e_dists: ') * ' ')
    start_string = pprint.pformat(start_dist_hat)

    # Check stop condition
    # Don't want to repeat calculations, so ith iterate checks previous update
    # For example, 0th iterate shows initial parameters, and 1st iterate shows results of first update
    ll = sum(ps)
    if ll0 is not None and abs(ll - ll0) < epsilon:
        print(f'FINAL VALUES')
        print('log-likelihood:', ll)
        print('delta log-likelihood:', ll - ll0 if ll0 is not None else None)
        print('t_dists:', t_string)
        print('e_dists:', e_string)
        print('start_dist:', start_string)
        break

    # Print results
    print(f'ITERATION {numiter}')
    print('log-likelihood:', ll)
    print('delta log-likelihood:', ll - ll0 if ll0 is not None else None)
    print('t_dists:', t_string)
    print('e_dists:', e_string)
    print('start_dist:', start_string)
    print()

    # Normalize all counts and update model
    t_dists_hat = {}
    for state1, t_count in t_counts.items():
        t_sum = sum(t_count.values())
        t_dist_hat = {}
        for state2, count in t_count.items():
            t_dist_hat[state2] = count / t_sum
        t_dists_hat[state1] = t_dist_hat

    e_dists_hat = {}
    for state, e_count in e_counts.items():
        e_sum = sum(e_count.values())
        e_dist_hat = {}
        for emit, count in e_count.items():
            e_dist_hat[emit] = count / e_sum
        e_dists_hat[state] = e_dist_hat

    start_sum = sum(start_count.values())
    start_dist_hat = {}
    for state, count in start_count.items():
        start_dist_hat[state] = count / start_sum

    ll0 = ll
    model_hat = HMM(t_dists=t_dists_hat, e_dists=e_dists_hat, start_dist=start_dist_hat)


# In[20]:
t_dists = {1: {1: 0.95, 2: 0.05},
           2: {1: 0.1, 2: 0.9}}
e_dists = {1: stats.norm(loc=-2.5, scale=25**0.5),  # Scale is standard deviation
           2: stats.gamma(a=1.5, scale=16**0.5)}
start_dist = {1: 0.5, 2: 0.5}

xs = np.linspace(-25, 25, 250)
ys1 = e_dists[1].pdf(xs)
ys2 = e_dists[2].pdf(xs)

plt.plot(xs, ys1, label='1')
plt.plot(xs, ys2, label='2')
plt.ylabel('Density')
plt.legend(**legend_kwargs)

plt.savefig('out/norm-gamma_pdf.png', dpi=dpi)
plt.savefig('out/norm-gamma_pdf.tiff', dpi=dpi)
plt.close()


# In[21]:
model = HMM(t_dists=t_dists, e_dists=e_dists, start_dist=start_dist)

data = [model.simulate(200, random_state=i) for i in range(10)]  # Use a different random seed for each example
data = [list(zip(*example)) for example in data]

xs, ys = data[0]
lines = {}
for state in t_dists:
    lines[state] = [x == state for x in xs]

fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained')
axs[0].plot(ys)
for state, line in sorted(lines.items()):
    axs[1].plot(line, label=state)
axs[1].set_xlabel('Time step')
axs[0].set_ylabel('Emission')
axs[1].set_ylabel('Label')
axs[1].legend(**legend_kwargs)

fig.savefig('out/norm-gamma_trace.png', dpi=dpi)
fig.savefig('out/norm-gamma_trace.tiff', dpi=dpi)
plt.close()


# In[22]:
xstack = np.stack([xs for xs, ys in data])
ystack = np.stack([ys for xs, ys in data])

# Make estimated transition distributions
t_dists_ML = {}
for state1, t_dist in t_dists.items():
    t_dist_ML = {}
    x1 = (xstack[:, :-1] == state1)
    x1_sum = x1.sum()
    for state2 in t_dist:
        x2 = (xstack[:, 1:] == state2)
        x12_sum = (x1 & x2).sum()
        t_dist_ML[state2] = x12_sum / x1_sum
    t_dists_ML[state1] = t_dist_ML
print('ML ESTIMATED T_DISTS')
for state, t_dist_ML in t_dists_ML.items():
    print(f'{state}: {t_dist_ML}')
print()

# Make estimated emission distributions
e_params_ML = {}
for state in e_dists:
    xs = xstack.ravel() == state
    ys = ystack.ravel()[xs]
    loc = ys.mean()
    scale = ys.var()
    e_params_ML[state] = {'mu': loc, 'sigma2': scale}
print('ML ESTIMATED E_PARAMS')
for state, e_param_ML in e_params_ML.items():
    print(f'{state}: {e_param_ML}')
print()

# Make estimated start distribution
start_dist_ML = {}
for state in start_dist:
    start_dist_ML[state] = (xstack[:, 0] == state).sum() / xstack.shape[0]
print('ML ESTIMATED START_DIST')
print(start_dist_ML)


# In[23]:
def norm_pdf(y, mu, sigma2):
    return stats.norm.pdf(y, loc=mu, scale=sigma2**0.5)


# In[24]:
e_funcs = {1: norm_pdf,
           2: norm_pdf}


# In[25]:
def norm_prime_mu(y, mu, sigma2, **kwargs):
    return (y - mu) / sigma2


def norm_prime_sigma2(y, mu, sigma2, **kwargs):
    term1 = - 1 / sigma2
    term2 = (y - mu) ** 2 / sigma2 ** 2
    return 0.5 * (term1 + term2) * sigma2


# In[26]:
e_primes = {1: {'mu': norm_prime_mu, 'sigma2': norm_prime_sigma2},
            2: {'mu': norm_prime_mu, 'sigma2': norm_prime_sigma2}}


# In[27]:
def mu2aux(mu, **kwargs):
    return mu


def aux2mu(mu_aux, **kwargs):
    return mu_aux


def sigma22aux(sigma2, **kwargs):
    return log(sigma2)


def aux2sigma2(sigma2_aux, **kwargs):
    return exp(sigma2_aux)


e_param2aux = {1: {'mu': mu2aux, 'sigma2': sigma22aux},
               2: {'mu': mu2aux, 'sigma2': sigma22aux}}
e_aux2param = {1: {'mu': aux2mu, 'sigma2': aux2sigma2},
               2: {'mu': aux2mu, 'sigma2': aux2sigma2}}


# In[28]:
params = fit_CML(data,
                 t_dists=t_dists_ML,
                 e_params=e_params_ML, e_funcs=e_funcs, e_primes=e_primes,
                 e_param2aux=e_param2aux, e_aux2param=e_aux2param,
                 start_dist=start_dist_ML, eta=0.3, maxiter=500, verbose=False)
t_dists_CML, e_params_CML, start_dist_CML = params

print('CML ESTIMATED T_DISTS')
for state, t_dist_CML in t_dists_CML.items():
    print(f'{state}: {t_dist_CML}')
print()
print('CML ESTIMATED E_PARAMS')
for state, e_param_CML in e_params_CML.items():
    print(f'{state}: {e_param_CML}')
print()
print('CML ESTIMATED START_DIST')
print(start_dist_CML)


# In[29]:
fig, axs = plt.subplots(2, 1, figsize=(6.4, 6), layout='constrained')
xs = np.linspace(-25, 25, 250)

ys1 = e_dists[1].pdf(xs)
ys2 = e_dists[2].pdf(xs)
for ax in axs:
    ax.plot(xs, ys1, label='true state 1', color='C0')
    ax.plot(xs, ys2, label='true state 2', color='C1')
    ax.set_ylabel('Density')

ys1 = stats.norm.pdf(xs, loc=e_params_ML[1]['mu'], scale=e_params_ML[1]['sigma2']**0.5)
ys2 = stats.norm.pdf(xs, loc=e_params_ML[2]['mu'], scale=e_params_ML[2]['sigma2']**0.5)
axs[0].plot(xs, ys1, label='ML state 1', color='C3')
axs[0].plot(xs, ys2, label='ML state 2', color='C2')
axs[0].legend(**legend_kwargs)

ys1 = stats.norm.pdf(xs, loc=e_params_CML[1]['mu'], scale=e_params_CML[1]['sigma2']**0.5)
ys2 = stats.norm.pdf(xs, loc=e_params_CML[2]['mu'], scale=e_params_CML[2]['sigma2']**0.5)
axs[1].plot(xs, ys1, label='CML state 1', color='C3')
axs[1].plot(xs, ys2, label='CML state 2', color='C2')
axs[1].legend(**legend_kwargs)

fig.savefig('out/norm-gamma_hat_pdf.png', dpi=dpi)
fig.savefig('out/norm-gamma_hat_pdf.tiff', dpi=dpi)
plt.close()

# In[30]:
e_dists_ML = {}
for state, e_param_ML in e_params_ML.items():
    e_dists_ML[state] = stats.norm(loc=e_param_ML['mu'], scale=e_param_ML['sigma2'] ** 0.5)
e_dists_CML = {}
for state, e_param_CML in e_params_CML.items():
    e_dists_CML[state] = stats.norm(loc=e_param_CML['mu'], scale=e_param_CML['sigma2'] ** 0.5)

model_ML = HMM(t_dists=t_dists_ML, e_dists=e_dists_ML, start_dist=start_dist_ML)
model_CML = HMM(t_dists=t_dists_CML, e_dists=e_dists_CML, start_dist=start_dist_CML)

xstack_ML = []
xstack_CML = []
for example in data:
    _, ys = example
    fbs_ML = model_ML.forward_backward(ys)
    fbs_CML = model_CML.forward_backward(ys)

    xstack_ML.append(fbs_ML[1])
    xstack_CML.append(fbs_CML[1])
xstack_ML = np.stack(xstack_ML)
xstack_CML = np.stack(xstack_CML)

threshold = 0.5
accuracy_ML = ((xstack_ML >= threshold) == (xstack == 1)).sum()
accuracy_CML = ((xstack_CML >= threshold) == (xstack == 1)).sum()
print('ML accuracy:', accuracy_ML / xstack.size)
print('CML accuracy:', accuracy_CML / xstack.size)


# In[31]:
fpr_ML, tpr_ML, _ = roc_curve((xstack == 1).ravel(), xstack_ML.ravel())
fpr_CML, tpr_CML, _ = roc_curve((xstack == 1).ravel(), xstack_CML.ravel())

fig, ax = plt.subplots(layout='constrained')
ax.plot(fpr_ML, tpr_ML, label='ML')
ax.plot(fpr_CML, tpr_CML, label='CML')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(**legend_kwargs)

idx_ML = fpr_ML <= 0.1
idx_CML = fpr_CML <= 0.1
axins = ax.inset_axes([0.25, 0.1, 0.7, 0.5])
axins.plot(fpr_ML[idx_ML], tpr_ML[idx_ML], label='ML')
axins.plot(fpr_CML[idx_CML], tpr_CML[idx_CML], label='CML')
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.spines[['right', 'top']].set_visible(True)
ax.indicate_inset_zoom(axins, edgecolor='black')

fig.savefig('out/norm-gamma_ROC.png', dpi=dpi)
fig.savefig('out/norm-gamma_ROC.tiff', dpi=dpi)
plt.close()
