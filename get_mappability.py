import sys
import numpy as np
import experiments
import pickle
from sklearn.cross_decomposition import CCA

DIM_LIM = 50000


def get_avg_cluster_loss(vectors, centers, square_loss=False, unitize=True):
    """
    :param vectors: the activations vectors to map onto the centroid vectors
    :param centers: the centroid vectors to map onto
    :param square_loss: whether to use square_loss norm and penalize the euclidian distance squared
    :param unitize: whether to unitize vectors
    :return: scalar-valued average and std square_loss norm reconstruction loss
    """

    # make so that each row is a point (neuron), and each column is a dimension (image)
    vectors = vectors.T
    centers = centers.T

    # get number of points for each
    v_num = np.shape(vectors)[0]
    c_num = np.shape(centers)[0]

    # reduce the number of samples if it's too big
    if np.shape(vectors)[1] > DIM_LIM:
        rand_idx = np.random.randint(0, np.shape(vectors)[1], size=DIM_LIM)
        vectors = vectors[:, rand_idx]
        centers = centers[:, rand_idx]

    if unitize:
        v_sums = np.sum(vectors, axis=1)
        c_sums = np.sum(centers, axis=1)
        vectors = vectors / v_sums[:, np.newaxis]
        centers = centers / c_sums[:, np.newaxis]
        vectors = np.nan_to_num(vectors)
        centers = np.nan_to_num(centers)

    losses = np.zeros((v_num,))

    # find closest cluster point square_loss norm for each point in vectors
    for i in range(v_num):
        best_loss = -1  # init at -1
        for j in range(c_num):
            dist = np.linalg.norm(vectors[i]-centers[j])
            if dist < best_loss or best_loss == -1:
                best_loss = dist
        if square_loss:
            losses[i] = best_loss**2
        else:
            losses[i] = best_loss

    return np.mean(losses), np.std(losses)


def get_avg_proj_loss(vectors, basis, square_loss=False, unitize=True):
    """
    :param vectors: the activations vectors to map onto the centroid vectors
    :param basis: the centroid vectors to map onto
    :param square_loss: whether to use square_loss norm and penalize the euclidian distance squared
    :param unitize: whether to unitize vectors
    :return: scalar-valued average square_loss norm reconstruction loss
    """

    vectors = vectors.T
    basis = basis.T

    # having too many dimensions will make the subspace projection much too data intensive, so we'll
    # reduce the number of samples if it's too big
    if np.shape(vectors)[1] > DIM_LIM:
        rand_idx = np.random.randint(0, np.shape(vectors)[1], size=DIM_LIM)
        vectors = vectors[:, rand_idx]
        basis = basis[:, rand_idx]

    if unitize:
        v_sums = np.sum(vectors, axis=1)
        b_sums = np.sum(basis, axis=1)
        vectors = vectors / v_sums[:, np.newaxis]
        basis = basis / b_sums[:, np.newaxis]
        vectors = np.nan_to_num(vectors)
        basis = np.nan_to_num(basis)

    v_num = np.shape(vectors)[0]

    m1 = np.matmul(basis, basis.T)
    m2 = np.linalg.pinv(m1)  # I use pinv because apparently sometimes the matrices are singular
    m3 = np.matmul(m2, basis)  # note that this is the space bottleneck step which creates a dim*dim matrix

    subspace_proj_mat = np.matmul(basis.T, m3)

    proj = np.matmul(vectors, subspace_proj_mat)

    if square_loss:
        losses = np.array([np.linalg.norm(vectors[i]-proj[i])**2 for i in range(v_num)])
    else:
        losses = np.array([np.linalg.norm(vectors[i] - proj[i]) for i in range(v_num)])

    return np.mean(losses), np.std(losses)


def get_avg_cca_loss(x, y):
    """
    :param x: the activations vectors to map onto
    :param y: the activations vectors to map onto x vectors
    :param square_loss: whether to use square_loss norm and penalize the euclidian distance squared
    :return: scalar-valued average rho from CCA
    """

    x = x.T
    y = y.T

    # reduce the number of samples if it's too big
    if np.shape(x)[1] > DIM_LIM:
        rand_idx = np.random.randint(0, np.shape(x)[1], size=DIM_LIM)
        x = x[:, rand_idx]
        y = y[:, rand_idx]

    # x_num = np.shape(x)[0]

    x = x.T
    y = y.T

    cca = CCA(max_iter=5000)

    cca.fit(x, y)

    r2 = cca.score(x, y)

    return r2

sizes = 5
size_idx_map = {0.25: 0, 0.5: 1, 1: 2, 2: 3, 4: 4}
regs = 3
layers = 4

# indexed[cluster_mean/cluster_std/proj_mean/proj_std/cca_mean/cca_std][size][unreg/allreg/random][layer]
unreg_results = np.zeros((6, sizes, regs, layers))
allreg_results = np.zeros((6, sizes, regs, layers))

large_nets = [86, 91]  # the 4x unregularized and allregularized nets
large_net_duplicates = {86: 232, 91: 233}  # the IDs for the same types of models but with a different random seed
opt_unreg = experiments.opt[large_nets[0]]
opt_allreg = experiments.opt[large_nets[1]]

# get all stored activations for the large nets
with open(opt_unreg.log_dir_base + opt_unreg.name + '/activations_test0' + '.pkl', 'rb') as f:
    unreg_acts = pickle.load(f)
with open(opt_allreg.log_dir_base + opt_allreg.name + '/activations_test0' + '.pkl', 'rb') as f:
    allreg_acts = pickle.load(f)

for ID in range(62, 92):

    if ID in large_nets:  # don't map nets onto themselves, map onto identically trained ones
        opt = experiments.opt[large_net_duplicates[ID]]
    else:
        opt = experiments.opt[ID]

    regs = 0
    if opt.hyper.augmentation:
        regs += 1
    if opt.hyper.drop_train < 1:
        regs += 1
    if opt.hyper.weight_decay > 0:
        regs += 1
    if regs == 1:
        continue  # if this isn't one of the nets we want to try mapping onto; we want regs \in [0, 3]

    # get positions to store the results in the result arrays
    size_idx = size_idx_map[opt.dnn.neuron_multiplier[0]]
    reg_idx = 0
    if regs == 3:
        reg_idx = 1
    elif opt.dataset.random_labels:
        reg_idx = 2

    print('Processing', opt.name)
    np.random.seed(opt.seed)

    # get activations for nets to map onto
    with open(opt.log_dir_base + opt.name + '/activations_test0' + '.pkl', 'rb') as f:
        opt_acts = pickle.load(f)

    # get and store average losses
    for l in range(layers):
        print('Layer:', l)
        # get mean and std square_loss losses
        # unreg_results[0, size_idx, reg_idx, l], unreg_results[1, size_idx, reg_idx, l] = \
        #     get_avg_cluster_loss(unreg_acts[l], opt_acts[l])
        # allreg_results[0, size_idx, reg_idx, l], allreg_results[1, size_idx, reg_idx, l] = \
        #     get_avg_cluster_loss(allreg_acts[l], opt_acts[l])
        # unreg_results[2, size_idx, reg_idx, l], unreg_results[3, size_idx, reg_idx, l] = \
        #     get_avg_proj_loss(unreg_acts[l], opt_acts[l])
        # allreg_results[2, size_idx, reg_idx, l], allreg_results[3, size_idx, reg_idx, l] = \
        #     get_avg_proj_loss(allreg_acts[l], opt_acts[l])
        unreg_results[4, size_idx, reg_idx, l] = get_avg_cca_loss(unreg_acts[l], opt_acts[l])
        allreg_results[4, size_idx, reg_idx, l] = get_avg_cca_loss(allreg_acts[l], opt_acts[l])
        sys.stdout.flush()

'''
1: sq loss not unitized
2: sq loss unitized
3: loss not unitized
4: loss unitized
5: cca
6: violin plots
'''

with open(opt_unreg.log_dir_base + opt_unreg.name + '/mappability5' + '.pkl', 'wb') as f:
    pickle.dump(unreg_results, f, protocol=2)
with open(opt_allreg.log_dir_base + opt_allreg.name + '/mappability5' + '.pkl', 'wb') as f:
    pickle.dump(allreg_results, f, protocol=2)

sys.stdout.flush()
print('get_mappability.py')
print(':)')
