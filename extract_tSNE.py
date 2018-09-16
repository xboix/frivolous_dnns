
from __future__ import print_function
import sys
import numpy as np
import os

import experiments_local as experiments

import pickle
import time

from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA


################################################################################################
# Read experiment to run
################################################################################################
ID_base = 2

opt = experiments.opt[ID_base]
with open(opt.log_dir_base + opt.name + '/activations0.pkl', 'rb') as f:
    a_base = pickle.load(f)

ID = 6

opt = experiments.opt[ID]
with open(opt.log_dir_base + opt.name + '/activations0.pkl', 'rb') as f:
    a = pickle.load(f)
with open(opt.log_dir_base + opt.name + '/selectivity0.pkl', 'rb') as f:
    s = pickle.load(f)

for LAYER in [3]:#range(4):
    res = a[LAYER].T
    tt = s[0][LAYER]
    res_base = a_base[LAYER].T

    fig, ax = plt.subplots(figsize=(7, 5))

    num_neurons = np.shape(res)[0]
    print(np.shape(res))
    for k in range(num_neurons):
        res[k, :] = res[k, :] / LA.norm(res[k, :], axis=0)

    num_neurons_base = np.shape(res_base)[0]
    for k in range(num_neurons_base):
        res_base[k, :] = res_base[k, :] / LA.norm(res_base[k, :], axis=0)

    KK = np.zeros([num_neurons, num_neurons_base])
    for k in range(num_neurons):
        for k_base in range(num_neurons_base):
            KK[k, k_base] = np.sum((res[k, :]-res_base[k_base, :])**2)


    tt = np.argmin(KK,axis=1)
    cc = np.random.rand(num_neurons_base, 3)

    print(np.shape(res))
    tsne = TSNE(n_components=2, verbose=1, perplexity=16, n_iter=300)
    tsne_results = tsne.fit_transform(res)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cc[tt], alpha=0.5)
    # plt.colorbar()
    plt.show()

    savepath = os.path.join('./', 'tSNE_' + str(ID) + '_' + str(LAYER) + '.pdf')
    plt.savefig(savepath, format='pdf', dpi=1000)
    #plt.close()




