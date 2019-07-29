from __future__ import print_function
import sys
import numpy as np
import os
import experiments
from data import cifar_dataset
from models import nets
import pickle
import time

# Read experiment to run
ID = int(sys.argv[1:][0])

opt = experiments.opt[ID]

# Skip execution if instructed in experiment
if opt.skip:
    print("SKIP")
    quit()

print('Experiment:', opt.name)

np.random.seed(opt.seed)


def pca(data, energy):
    # retuerns eig vals and vecs as well as the number of pc's needed to capture proportions of variance
    data = data - data.mean(axis=0)
    covariance = np.cov(data, rowvar=False)
    covariance = np.nan_to_num(covariance)  # I added this to fix one problem that one matrix was causing in one opt
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # eigenvals are sorted small to large

    th = np.zeros(len(energy))
    total = np.sum(eigenvalues)
    for en, idx_en in zip(energy, range(len(energy))):
        accum = 0
        k = 1
        while accum <= en:
            accum += eigenvalues[-k] / total
            k += 1
        th[idx_en] = k - 1  # th is num of eigenvectors needed to explain en proportion of variance
    return th, eigenvalues, eigenvectors


def get_compressability95(eigenvalues):
    # returns what proportion of eigenvals needed to represent 0.95 of the variation
    total = np.sum(eigenvalues)
    accum = 0
    k = 0
    while accum <= 0.05:
        accum += eigenvalues[k] / total
        k += 1
    return k / len(eigenvalues)


def selectivity(res, gt, res_test, gt_test):  # res is activations, gt is labels
    # for each neuron returning max argcategory of cat_avg-noncat_avg / cat_avg+noncat_avg
    # ranges from 0 to 1 where a higher sel coeff means a higher level of selectivity
    num_neurons = np.shape(np.mean(res, axis=0))[0]
    ave_c = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    ave_c_test = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    ave_all = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    ave_all_test = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    for k in np.unique(gt, axis=0).tolist():  # for each category
        ave_c[:, k] = np.mean(res[gt == k], axis=0)
        ave_c_test[:, k] = np.mean(res_test[gt_test == k], axis=0)
        ave_all[:, k] = np.mean(res[gt != k], axis=0)
        ave_all_test[:, k] = np.mean(res_test[gt_test != k], axis=0)

    idx_max = np.argmax(ave_c, axis=1)
    sel = np.zeros([num_neurons])
    sel_test = np.zeros([num_neurons])

    for idx_k, k in enumerate(idx_max.tolist()):  # for each neuron and its idx_max
        sel[idx_k] = (ave_c[idx_k, k] - ave_all[idx_k, k]) / (ave_c[idx_k, k] + ave_all[idx_k, k])
        sel_test[idx_k] = (ave_c_test[idx_k, k] - ave_all_test[idx_k, k]) / (
                    ave_c_test[idx_k, k] + ave_all_test[idx_k, k])

    sel_gen = ((sel - sel_test) ** 2)  # will always be positive, measures how different train/test sel is

    return sel, sel_test, sel_gen


def get_corr(res, res_test):  # res is activations
    # returns num_neurons x num_neurons correlation matrix
    # make so that each row is a neuron, so each point is a n_example dimensional neuron vector
    res_t = res.T
    res_test_t = res_test.T
    r = np.corrcoef(res_t)
    r_test = np.corrcoef(res_test_t)
    return r, r_test


def get_similarity(r, threshold=0.5):
    # returns average number of similar neurons per neuron where |r| >= threshold
    num_similar = np.sum(np.abs(r) >= threshold, axis=0) - 1
    return np.mean(num_similar), np.std(num_similar)


t0 = time.time()
if not os.path.isfile(opt.log_dir_base + opt.name + '/activations0.pkl'):
    print("Error: can't find needed files in dir ", opt.log_dir_base + opt.name)
    sys.exit()

t0 = time.time()

for cross in range(3):
    # results indexed [num_components/evals/evecs/acc/layer_nodes/num_components][train/test]
    results = [[[] for i in range(2)] for i in range(6)]

    with open(opt.log_dir_base + opt.name + '/activations' + str(cross) + '.pkl', 'rb') as f:
        res = pickle.load(f)
    with open(opt.log_dir_base + opt.name + '/labels' + str(cross) + '.pkl', 'rb') as f:
        gt_labels = pickle.load(f)
    with open(opt.log_dir_base + opt.name + '/accuracy' + str(cross) + '.pkl', 'rb') as f:
        acc = pickle.load(f)
    with open(opt.log_dir_base + opt.name + '/activations_test' + str(cross) + '.pkl', 'rb') as f:
        res_test = pickle.load(f)
    with open(opt.log_dir_base + opt.name + '/labels_test' + str(cross) + '.pkl', 'rb') as f:
        gt_labels_test = pickle.load(f)
    with open(opt.log_dir_base + opt.name + '/accuracy_test' + str(cross) + '.pkl', 'rb') as f:
        acc_test = pickle.load(f)

    corr = []
    corr_test = []
    similarity = [[[] for i in range(2)] for j in range(2)]  # will be indexed by [train/test][mean/std][layer]
    for layer in range(len(res)):  # get r, r_test, and the mean and std for similarity
        corr_tmp, corr_test_tmp = get_corr(res[layer], res_test[layer])
        corr.append(corr_tmp)
        corr_test.append(corr_test_tmp)

        mean_sim, std_sim = get_similarity(corr_tmp)
        similarity[0][0].append(mean_sim)
        similarity[0][1].append(std_sim)

        mean_sim, std_sim = get_similarity(corr_test_tmp)
        similarity[1][0].append(mean_sim)
        similarity[1][1].append(std_sim)

    with open(opt.log_dir_base + opt.name + '/corr' + str(cross) + '.pkl', 'wb') as f:
        pickle.dump(corr, f, protocol=2)

    with open(opt.log_dir_base + opt.name + '/corr_test' + str(cross) + '.pkl', 'wb') as f:
        pickle.dump(corr_test, f, protocol=2)

    with open(opt.log_dir_base + opt.name + '/similarity' + str(cross) + '.pkl', 'wb') as f:
        pickle.dump(similarity, f, protocol=2)

    for layer in range(len(res)):
        num_components, evals, evecs = pca(res[layer], (0.95, 0.8))
        results[0][0].append(num_components)
        results[1][0].append(evals)
        results[2][0].append(evecs)
        results[3][0].append(acc)
        results[4][0].append(np.shape(res[layer])[-1])
        compressability_95 = get_compressability95(evals)
        results[5][0].append(compressability_95)

    for layer in range(len(res_test)):
        num_components, evals, evecs = pca(res_test[layer], (0.95, 0.8))
        results[0][1].append(num_components)
        results[1][1].append(evals)
        results[2][1].append(evecs)
        results[3][1].append(acc_test)
        results[4][1].append(np.shape(res_test[layer])[-1])
        compressability_95 = get_compressability95(evals)
        results[5][1].append(compressability_95)

    results_selectivity = [[] for i in range(3)]  # [sel/sel_test/sel_gen][layer][neuron]
    for layer in range(len(res)):
        sel, sel_test, sel_gen = selectivity(res[layer], gt_labels[layer], res_test[layer], gt_labels_test[layer])
        results_selectivity[0].append(sel)
        results_selectivity[1].append(sel_test)
        results_selectivity[2].append(sel_gen)

    with open(opt.log_dir_base + opt.name + '/redundancy' + str(cross) + '.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)

    with open(opt.log_dir_base + opt.name + '/selectivity' + str(cross) + '.pkl', 'wb') as f:
        pickle.dump(results_selectivity, f, protocol=2)

sys.stdout.flush()
t1 = time.time()
print('Time: ', t1 - t0)
print('get_redundancy.py')
print(':)')
