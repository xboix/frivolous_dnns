import os.path
import shutil
import sys
import os
import numpy as np
import gc
import tensorflow as tf
from nets import nets
from data import data
from runs import preprocessing
import pickle


def pca(data, energy=(0.95,)):
    # retuerns eig vals and vecs as well as the number of pc's needed to capture proportions of variance
    data = data - data.mean(axis=0)
    covariance = np.cov(data, rowvar=False)
    covariance = np.nan_to_num(covariance)  # I added this to fix one problem that one matrix was causing in one opt
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # eigenvals are sorted small to large

    en_evecs = np.zeros(len(energy))
    total = np.sum(eigenvalues)
    for en, idx_en in zip(energy, range(len(energy))):
        accum = 0
        k = 1
        while accum <= en:
            accum += eigenvalues[-k] / total
            k += 1
        en_evecs[idx_en] = k - 1  # en_evecs is num of eigenvectors needed to explain en proportion of variance

    total = np.sum(eigenvalues)
    accum = 0
    k = 0
    while accum <= 0.05:
        accum += eigenvalues[k] / total
        k += 1
    compressability = k / len(eigenvalues)

    return compressability, en_evecs, eigenvalues, eigenvectors


def get_selectivity(res, gt):  # res is activations, gt is labels
    # for each neuron returning max argcategory of cat_avg-noncat_avg / cat_avg+noncat_avg
    # ranges from 0 to 1 where a higher sel coeff means a higher level of selectivity
    num_neurons = np.shape(np.mean(res, axis=0))[0]
    ave_c = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    ave_all = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    for k in np.unique(gt, axis=0).tolist():  # for each category
        ave_c[:, k-1] = np.mean(res[gt == k], axis=0)  # there is a k-1 because the labels are 1-1000 not 0-999
        ave_all[:, k-1] = np.mean(res[gt != k], axis=0)  # there is a k-1 because the labels are 1-1000 not 0-999

    idx_max = np.argmax(ave_c, axis=1)
    sel = np.zeros([num_neurons])

    for idx_k, k in enumerate(idx_max.tolist()):  # for each neuron and its idx_max
        sel[idx_k] = (ave_c[idx_k, k] - ave_all[idx_k, k]) / (ave_c[idx_k, k] + ave_all[idx_k, k])

    return sel


def get_corr(res):  # res is activations
    # returns num_neurons x num_neurons correlation matrix
    # make so that each row is a neuron, so each point is a n_example dimensional neuron vector
    res_t = res.T
    r = np.corrcoef(res_t)
    return r


def get_similarity(r, threshold=0.5):
    # returns average number of similar neurons per neuron where |r| >= threshold
    num_similar = np.sum(np.abs(r) >= threshold, axis=0) - 1
    return np.mean(num_similar), np.std(num_similar)


def run(opt):

    # Skip execution if instructed in experiment
    if opt.skip:
        print("SKIP")
        quit()

    print('name:', opt.name)
    print('factor:', opt.dnn.factor)
    print('batch size:', opt.hyper.batch_size)

    for cross in range(3):

        print('cross', cross)

        with open(opt.results_dir + opt.name + '/activations' + str(cross) + '.pkl', 'rb') as f:
            res = pickle.load(f)
        with open(opt.results_dir + opt.name + '/labels' + str(cross) + '.pkl', 'rb') as f:
            gt_labels = pickle.load(f)

        num_layers = len(res)

        corr = []
        similarity = []
        compressability = []
        selectivity = []

        for layer in range(num_layers):

            print('processing_layer:', str(layer+1) + '/' + str(num_layers))
            sys.stdout.flush()

            corr_layer = get_corr(res[layer])
            corr.append(corr_layer)
            sim_mean, sim_std = get_similarity(corr_layer)
            similarity.append([sim_mean, sim_std])
            compress_95, _, _, _ = pca(res[layer])
            compressability.append(compress_95)
            sel = get_selectivity(res[layer], gt_labels[layer])
            selectivity.append(sel)

        print('now writing .pkl files...')

        with open(opt.results_dir + opt.name + '/corr' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(corr, f, protocol=2)
        with open(opt.results_dir + opt.name + '/similarity' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(similarity, f, protocol=2)
        with open(opt.results_dir + opt.name + '/compressability' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(compressability, f, protocol=2)
        with open(opt.results_dir + opt.name + '/selectivity' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(selectivity, f, protocol=2)

        tf.reset_default_graph()
        gc.collect()
        sys.stdout.flush()

    print(":)")
