from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
import experiments
from data import cifar_dataset
from models import nets
import os
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

NUM_TYPES = 5
# perturbation types: 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko

np.random.seed(opt.seed)


def create_graph(perturbation_type):
    ################################################################################################
    # Define training and validation datasets thorugh Dataset API
    ################################################################################################

    with open(opt.log_dir_base + opt.name + '/selectivity0.pkl', 'rb') as f:
        selectivity = pickle.load(f)  # [sel, sel_test, sel_gen][layer][neuron]

    # Initialize dataset and creates TF records if they do not exist
    # Initialize dataset and creates TF records if they do not exist
    if opt.dataset_name == 'cifar':
        from data import cifar_dataset
        dataset = cifar_dataset.Cifar10(opt)
    elif opt.dataset_name == 'rand10':
        from data import rand10_dataset
        dataset = rand10_dataset.Rand10(opt)
    elif opt.dataset_name == 'rand100':
        from data import rand100_dataset
        dataset = rand100_dataset.Rand100(opt)
    elif opt.dataset_name == 'rand1000':
        from data import rand1000_dataset
        dataset = rand1000_dataset.Rand1000(opt)
    elif opt.dataset_name == 'rand10000':
        from data import rand10000_dataset
        dataset = rand10000_dataset.Rand10000(opt)
    elif opt.dataset_name == 'rand100000':
        from data import rand100000_dataset
        dataset = rand100000_dataset.Rand100000(opt)

    # No repeatable dataset for testing
    train_dataset_full = dataset.create_dataset(augmentation=False, standarization=True, set_name='train', repeat=False)
    val_dataset_full = dataset.create_dataset(augmentation=False, standarization=True, set_name='test', repeat=False)
    test_dataset_full = dataset.create_dataset(augmentation=False, standarization=True, set_name='test', repeat=False)

    # Hadles to switch datasets
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.compat.v1.data.Iterator.from_string_handle(handle, val_dataset_full.output_types,
                                                   val_dataset_full.output_shapes)

    train_iterator_full = train_dataset_full.make_initializable_iterator()
    val_iterator_full = val_dataset_full.make_initializable_iterator()
    test_iterator_full = test_dataset_full.make_initializable_iterator()
    ################################################################################################

    ################################################################################################
    # DNN
    ################################################################################################

    # Get data from dataset
    image, y_ = iterator.get_next()

    if opt.dataset_name == 'cifar':
        image = tf.image.resize_images(image, [opt.hyper.image_size, opt.hyper.image_size])
        if opt.extense_summary:
            tf.summary.image('input', image)
    elif opt.dataset_name == 'rand10':
        image = tf.compat.v1.reshape(image, [-1, 10])
    else:
        image = tf.compat.v1.reshape(image, [-1, 10000])
    # Call DNN
    dropout_rate = tf.compat.v1.placeholder(tf.float32)

    select = [tf.compat.v1.placeholder(tf.float32, shape=(len(selectivity[1][k]))) for k in range(len(selectivity[1]))]
    # select in THIS case is indexed [layer][neuron]

    perturbation_params = tf.compat.v1.placeholder(tf.float32, shape=[NUM_TYPES, opt.dnn.layers])  # idx: [type][layer]

    to_call = getattr(nets, opt.dnn.name + "_test")
    y, _ = to_call(image, dropout_rate, select, opt, dataset.list_labels, perturbation_params, perturbation_type)

    # Accuracy
    gt = y_
    y = tf.argmax(y, 1)
    im_prediction = tf.equal(y, gt)
    im_prediction = tf.cast(im_prediction, tf.float32)
    accuracy = tf.reduce_mean(im_prediction)

    num_iter_train = int(dataset.num_images_training * opt.dataset.proportion_training_set / opt.hyper.batch_size) - 1
    num_iter_test = int(dataset.num_images_test / opt.hyper.batch_size) - 1
    num_iter_val = int(dataset.num_images_val / opt.hyper.batch_size) - 1

    return accuracy, y, gt, im_prediction, handle, dropout_rate, perturbation_params, select, \
           train_iterator_full, val_iterator_full, test_iterator_full, num_iter_train, num_iter_val, num_iter_test
    ################################################################################################


def test_robustness(handle, dropout_rate, perturbation_params, select, opt, range_robustness, iterator_dataset,
                    handle_dataset, num_iter, ptype, cross):
    # perturbation_params indexed [type][layer]
    # select indexed [layer][neuron]

    with open(opt.log_dir_base + opt.name + '/corr' + str(cross) + '.pkl', 'rb') as f:  # only do for the first cross
        corr = pickle.load(f)

    corr = [np.abs(corr[k]) for k in range(len(corr))]  # take the absolute values
    corr_single = [corr[k][np.random.randint(0, np.shape(corr[k])[0])] for k in range(len(corr))]
    p_order = [np.flip(np.argsort(corr_single[k]), axis=0) for k in range(len(corr_single))]
    # p_order[layer] is now an np array of indices for neurons sorted from high abs(corr) to low
    rand_p_order = [np.copy(p_order[k]) for k in range(len(p_order))]
    for k in range(len(rand_p_order)):
        np.random.shuffle(rand_p_order[k])

    to_perturb = [np.zeros([len(p_order[k])]) for k in range(len(p_order))]  # indexed [layer][neuron]

    results_robustness = np.zeros([opt.dnn.layers + 1, len(range_robustness), 3])  # idx: [layer][range][max/ave/min]

    outputs_base = np.array([])
    test_rob = np.zeros([NUM_TYPES, opt.dnn.layers])  # initially, it's zeros, so there will be no perturbation
    sess.run(iterator_dataset.initializer)
    for _ in range(num_iter):  # fills im_pred_gt with predictions and im_y with labels
        dall = {}
        dall.update({i: d for i, d in zip(select, to_perturb)})  # i is a list of neurons, and d is node binaries
        dall.update({handle: handle_dataset, dropout_rate: opt.hyper.drop_test, perturbation_params: test_rob})
        acc_tmp, y_iter, gt_iter, im_pred_iter = sess.run([accuracy, y, gt, im_prediction], feed_dict=dall)
        outputs_base = np.concatenate((outputs_base, y_iter))

    sys.stdout.flush()

    for layer in range(opt.dnn.layers + 1):  # +1 is for doing all layers together

        # This is only okay to do if were doing activation perturbations and not weight perturbations
        # b/c the last layer isn't one where we ever apply activation perturbations for obvious reasons
        if layer == (opt.dnn.layers-1):
            continue

        print('Processing Layer:', layer)

        for noise_id, noise_level in enumerate(range_robustness):
            sess.run(iterator_dataset.initializer)

            test_rob = np.zeros([NUM_TYPES, opt.dnn.layers])  # indexed [perturbation][layer]
            to_perturb = [np.zeros([len(p_order[k])]) for k in range(len(p_order))]  # indexed [layer][neuron]

            # this if/else block fills in to_perturb with min/max values for selectivity to use.
            # ptypes: 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko
            if layer < opt.dnn.layers:  # if doing just a layer
                if ptype == 4:  # targeted ko
                    min_idx = 0
                    max_idx = int(noise_level * len(p_order[layer]))
                    to_perturb[layer][p_order[layer][min_idx:max_idx]] = 1.0  # get the most similar neurons
                    test_rob[ptype, layer] = 1.0
                elif ptype == 2:  # random ko
                    min_idx = 0
                    max_idx = int(noise_level * len(p_order[layer]))
                    to_perturb[layer][rand_p_order[layer][min_idx:max_idx]] = 1.0  # get random neurons
                    test_rob[4, layer] = 1.0  # the idx is 4 instead of ptype b/c we are using a random mask
                else:
                    test_rob[ptype, layer] = noise_level

            else:  # if doing the whole network
                if ptype == 4:  # targeted ko
                    for l in range(len(p_order)):  # for each layer
                        min_idx = 0
                        max_idx = int(noise_level * len(p_order[l]))
                        to_perturb[l][p_order[l][min_idx:max_idx]] = 1.0
                    test_rob[ptype, :] = 1.0
                elif ptype == 2:  # random ko
                    for l in range(len(p_order)):  # for each layer
                        min_idx = 0
                        max_idx = int(noise_level * len(p_order[l]))
                        to_perturb[l][rand_p_order[l][min_idx:max_idx]] = 1.0
                    test_rob[4, :] = 1.0  # the idx is 4 instead of ptype b/c we are using a random mask
                else:
                    test_rob[ptype, :] = noise_level

            sess.run(iterator_dataset.initializer)

            output_perturbation = np.array([])
            for idx in range(num_iter):
                # runs the model for an iteration with the to_perturb perturbations
                dall = {}
                dall.update({i: d for i, d in zip(select, to_perturb)})  # i is neuron list, and d is node binaries
                dall.update({handle: handle_dataset, dropout_rate: opt.hyper.drop_test, perturbation_params: test_rob})
                acc_tmp, y_iter, gt_iter, im_pred_iter = sess.run([accuracy, y, gt, im_prediction], feed_dict=dall)
                output_perturbation = np.concatenate((output_perturbation, y_iter))  # perturb results

            prop_same_label = np.mean(outputs_base == output_perturbation)  # proportion of same prediections
            results_robustness[layer, noise_id, :] = prop_same_label

        sys.stdout.flush()

    return results_robustness  # indexed [layer][noise_id][max/ave/min]


t0 = time.time()

# For std
for cross in range(3):

    print('Cross:', cross)

    range_len = 7
    knockout_idx = [1, 2, 4]
    noise_idx = [0, 3]
    knockout_range = np.linspace(0.0, 1.0, num=range_len, endpoint=True)
    noise_range = np.linspace(0.0, 1.0, num=range_len, endpoint=True)
    # multiplicative_range = np.arange(0.0, 1.2, 0.2)

    results = [[[np.zeros([opt.dnn.layers + 1, range_len, 3])] for j in range(2)] for i in range(NUM_TYPES)]

    for ptype in range(NUM_TYPES):
        # perturbation types: 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko
        if ptype in [0, 1, 3, 4]:  # for now, we skip over weight perturbations and only look at activation ones
            continue

        print("Perturbation type: " + str(ptype))
        accuracy, y, gt, im_prediction, handle, dropout_rate, perturbation_params, select, train_iterator_full, \
        val_iterator_full, test_iterator_full, num_iter_train, num_iter_val, num_iter_test = create_graph(ptype)
        # note that HERE ^ perturbation_params is indexed [type][layer] select is indexed [layer][neuron]

        # allow for GPU memory to be allocated as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Set up checkpoint
            saver = tf.compat.v1.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)
            print("RESTORE")
            if os.path.isdir(opt.log_dir_base + opt.name + '/models/'):
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
            elif os.path.isdir(opt.log_dir_base + opt.name + '/'):
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/'))
            else:
                print('Can\'t find model save path.')
                quit()

            # Run test
            train_handle_full = sess.run(train_iterator_full.string_handle())
            validation_handle_full = sess.run(val_iterator_full.string_handle())
            test_handle_full = sess.run(test_iterator_full.string_handle())

            if ptype in knockout_idx:
                range_robustness = knockout_range
            else:
                range_robustness = noise_range

            # results is indexed [ptype][train/test][layer][range_robustness][max/ave/min]
            # perturbation_params indexed [type][layer]
            # select indexed [layer][neuron]
            print('Processing train set')
            results[ptype][0][:] = test_robustness(handle, dropout_rate, perturbation_params, select, opt,
                                                   range_robustness, train_iterator_full, train_handle_full,
                                                   num_iter_train, ptype, cross)
            print('Processing test set')
            results[ptype][1][:] = test_robustness(handle, dropout_rate, perturbation_params, select, opt,
                                                   range_robustness, test_iterator_full, test_handle_full,
                                                   num_iter_test, ptype, cross)

        tf.reset_default_graph()

    with open(opt.log_dir_base + opt.name + '/robustness' + str(cross) + '.pkl', 'wb') as f:
        pickle.dump(results, f)

sys.stdout.flush()
print('Total time: ', time.time() - t0)
print('get_robustness.py')
print(':)')
