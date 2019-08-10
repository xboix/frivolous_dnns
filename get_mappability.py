import sys
import numpy as np
import experiments
import pickle
from sklearn.cross_decomposition import CCA

# Note: give this a few days (3?) to run on om

DIM_LIM = 50000


def get_cluster_results(vectors, centers, square_loss=False, unitize=True):
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


def get_corr_results(vectors, centers):
    """
    :param vectors: the activations vectors to map onto the centroid vectors
    :param centers: the centroid vectors to map onto
    :return: scalar-valued average and std best r coeff for the neurons in vectors
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

    corrs = np.zeros((v_num,))

    # find the best r for each neuron in vectors with one in centers
    for i in range(v_num):
        best_r = -1  # init at -1
        for j in range(c_num):
            abs_r = np.abs(np.corrcoef(vectors[i], centers[j]))[0, 1]
            if abs_r > best_r:
                best_r = abs_r
        corrs[i] = best_r

    return np.mean(corrs), np.std(corrs)


def get_proj_results(vectors, basis, square_loss=False, unitize=True):
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


def get_cca_r2(y, x):
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


num_sizes = 5
size_idx_map = {0.25: 0, 0.5: 1, 1: 2, 2: 3, 4: 4}
num_regs = 3  # unreg, allreg, random
num_layers = 4
results_dir = '/om/user/scasper/workspace/models/replication/'

################################################################
results = np.zeros((4, 7, num_sizes, num_regs, num_layers))

# index of first dimension:
# 0: small_unreg net, ID62
# 1: small_allreg net, ID67
# 2: large_unreg net, ID86
# 3: large_allreg net, ID91

# index of second dimension:
# 0: cluster mean loss
# 1: cluster std loss
# 2: best_r mean
# 3: best_r std
# 4: ssp mean loss
# 5: ssp std loss
# 6: cca r^2 (which is not analogous to the pearson r^2)
################################################################

net_ids = [62, 67, 86, 91]
net_duplicates_map = {62: 239, 67: 240, 86: 232, 91: 233}  # IDs for same model but with diff random seed

for results_id, from_id in zip(range(4), net_ids):  # for each, map/score the activations FROM it ONTO the others

    print('\nprocessing net with id', from_id)

    from_opt = experiments.opt[from_id]

    # get stored TEST activations
    with open(from_opt.log_dir_base + from_opt.name + '/activations_test0' + '.pkl', 'rb') as f:
        from_acts = pickle.load(f)

    for onto_id in range(62, 92):

        if from_id == onto_id:  # don't map nets onto themselves, map onto identically trained ones
            onto_opt = experiments.opt[net_duplicates_map[onto_id]]
        else:
            onto_opt = experiments.opt[onto_id]

        num_regs = 0
        if onto_opt.hyper.augmentation:
            num_regs += 1
        if onto_opt.hyper.drop_train < 1:
            num_regs += 1
        if onto_opt.hyper.weight_decay > 0:
            num_regs += 1
        if num_regs == 1:
            continue  # if this isn't one of the nets we want to try mapping onto; we want num_regs \in [0, 3]

        # get positions to store the results in the result arrays
        size_idx = size_idx_map[onto_opt.dnn.neuron_multiplier[0]]
        reg_idx = 0
        if num_regs == 3:
            reg_idx = 1
        elif onto_opt.dataset.random_labels:
            reg_idx = 2

        print('mapping/scoring onto', onto_opt.name)

        np.random.seed(onto_opt.seed)

        # get TEST activations for net to map onto
        with open(onto_opt.log_dir_base + onto_opt.name + '/activations_test0' + '.pkl', 'rb') as f:
            onto_acts = pickle.load(f)

        for layer in range(num_layers):
            print('processing layer,', layer)

            results[results_id, 0, size_idx, reg_idx, layer], results[results_id, 1, size_idx, reg_idx, layer] = \
                get_cluster_results(from_acts[layer], onto_acts[layer])
            results[results_id, 2, size_idx, reg_idx, layer], results[results_id, 3, size_idx, reg_idx, layer] = \
                get_corr_results(from_acts[layer], onto_acts[layer])
            results[results_id, 4, size_idx, reg_idx, layer], results[results_id, 5, size_idx, reg_idx, layer] = \
                get_proj_results(from_acts[layer], onto_acts[layer])
            results[results_id, 6, size_idx, reg_idx, layer] = \
                get_cca_r2(from_acts[layer], onto_acts[layer])

            sys.stdout.flush()

with open(results_dir + '.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=2)

sys.stdout.flush()
print('get_mappability.py')
print(':)')
