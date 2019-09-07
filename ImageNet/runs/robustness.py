import os.path
import shutil
import sys
import numpy as np
import gc
import tensorflow as tf
from nets import nets
from data import data
from runs import preprocessing
import pickle

NUM_PTYPES = 5


def create_graph(opt, ptype):

    # I found that this was necessary in order to get everything to git on the om gpus
    # This doesn't affect anything because this is just for evaluation
    # It's ad hoc, but it works well
    if opt.dnn.name == 'resnet':
        if opt.hyper.batch_size * opt.dnn.factor >= 3072:
            opt.hyper.batch_size = int(opt.hyper.batch_size / 4)
        elif opt.hyper.batch_size * opt.dnn.factor >= 2048:
            opt.hyper.batch_size = int(opt.hyper.batch_size / 2)
    elif opt.dnn.name == 'inception':
        opt.hyper.batch_size = 256

    with open(opt.results_dir + opt.name + '/selectivity.pkl', 'rb') as f:
        selectivity = pickle.load(f)  # indexed [layer][neuron]

    select = [tf.compat.v1.placeholder(tf.float32, shape=(len(selectivity[k]))) for k in range(len(selectivity))]
    # select in THIS case is indexed [layer][neuron]

    perturbation_params = tf.compat.v1.placeholder(tf.float32, shape=[NUM_PTYPES, opt.dnn.layers])  # [type][layer]

    # Initialize dataset and creates TF records if they do not exist
    dataset = data.ImagenetDataset(opt)

    # Repeatable datasets for training
    val_dataset = dataset.create_dataset(set_name='val', repeat=True)

    # Handles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, val_dataset.output_types, val_dataset.output_shapes)

    val_iterator = val_dataset.make_initializable_iterator()

    num_iter_val = int(dataset.num_total_images / opt.hyper.batch_size) + 1
    ################################################################################################

    ################################################################################################
    # Declare DNN
    ################################################################################################

    # Get data from dataset dataset
    image, label = iterator.get_next()

    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']

    if opt.dnn.name == 'resnet':
        image_split = [tf.reshape(t, [-1, data._DEFAULT_IMAGE_SIZE, data._DEFAULT_IMAGE_SIZE, 3]) for t in
                       zip(tf.split(image, len(gpus)))]

    elif opt.dnn.name == 'inception':
        image = tf.cast(image, tf.float32)
        image_split = [tf.reshape(t, [-1, 299, 299, 3]) for t in
                       zip(tf.split(image, len(gpus)))]

    label_split = [tf.reshape(t, [-1]) for t in zip(tf.split(label, len(gpus)))]

    # Call DNN
    logits_list = []
    activations_list = []
    for idx_gpu, gpu in enumerate(gpus):

        with tf.device(gpu):
            with tf.name_scope('gpu_' + str(idx_gpu)) as scope:
                to_call = getattr(preprocessing, opt.dnn.name)
                u_images = []
                for _image in tf.unstack(image_split[idx_gpu], num=opt.hyper.batch_size / len(gpus), axis=0):
                    im_tmp = to_call(_image, opt)
                    u_images.append(im_tmp)
                _images = tf.stack(u_images)

                to_call = getattr(nets, opt.dnn.name + '_test')  # _test appended to name for robustness testing

                logit = to_call(_images, opt, select, perturbation_params, ptype, idx_gpu)

                tf.get_variable_scope().reuse_variables()
                logits_list.append(logit)

    logits = tf.reshape(tf.stack(logits_list, axis=0), [-1, 1001])

    pred_label = tf.argmax(logits, axis=1)
    acc_1 = tf.nn.in_top_k(predictions=logits, targets=label, k=1, name='top_1_op')
    acc_5 = tf.nn.in_top_k(predictions=logits, targets=label, k=5, name='top_5_op')
    ################################################################################################

    config = tf.ConfigProto(allow_soft_placement=True)  # inter_op_parallelism_threads=80,
                            #intra_op_parallelism_threads=80,
                            #
    #config.gpu_options.allow_growth = True

    return acc_1, acc_5, pred_label, label, handle, perturbation_params, select, val_iterator, num_iter_val


def test_robustness(sess, pred_label, handle, perturbation_params, select, opt, range_robustness,
                    val_iterator, val_handle_full, num_iter_val, ptype):

    results_robustness = np.zeros([opt.dnn.layers + 1, len(range_robustness)])  # idx: [layer][perturb_amount]

    with open(opt.results_dir + opt.name + '/corr.pkl', 'rb') as f:  # only do for the first cross
        corr = pickle.load(f)
    corr = [np.abs(corr[k]) for k in range(len(corr))]  # take the absolute values
    corr_single = [corr[k][np.random.randint(0, np.shape(corr[k])[0])] for k in range(len(corr))]
    p_order = [np.flip(np.argsort(corr_single[k]), axis=0) for k in range(len(corr_single))]
    # p_order[layer] is now an np array of indices for neurons sorted from high abs(corr) to low
    rand_p_order = [np.copy(p_order[k]) for k in range(len(p_order))]
    for k in range(len(rand_p_order)):
        np.random.shuffle(rand_p_order[k])

    to_perturb = [np.zeros([len(p_order[k])]) for k in range(len(p_order))]  # indexed [layer][neuron]

    outputs_base = np.array([])
    test_rob = np.zeros([NUM_PTYPES, opt.dnn.layers])  # initially, it's zeros, so there will be no perturbation
    sess.run(val_iterator.initializer)
    for _ in range(num_iter_val):  # fills im_pred_gt with predictions and im_y with labels
        dall = {}
        dall.update(
            {i: d for i, d in zip(select, to_perturb)})  # i is a list of neurons, and d is node binaries
        dall.update({handle: val_handle_full, perturbation_params: test_rob})
        base_pred_label = sess.run(pred_label, feed_dict=dall)
        outputs_base = np.concatenate((outputs_base, base_pred_label))

    sys.stdout.flush()

    for layer in range(opt.dnn.layers + 1):  # +1 is for doing all layers together

        print('processing layer: ' + str(layer+1) + '/' + str(opt.dnn.layers))

        for noise_id, noise_level in enumerate(range_robustness):
            print('processing perturbation amount:', noise_level)
            sess.run(val_iterator.initializer)

            test_rob = np.zeros([NUM_PTYPES, opt.dnn.layers])  # indexed [perturbation][layer]
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

            sess.run(val_iterator.initializer)

            output_perturbation = np.array([])
            for idx in range(num_iter_val):
                # runs the model for an iteration with the to_perturb perturbations
                dall = {}
                dall.update(
                    {i: d for i, d in zip(select, to_perturb)})  # i is neuron list, and d is node binaries
                dall.update({handle: val_handle_full, perturbation_params: test_rob})
                perturb_pred_label = sess.run(pred_label, feed_dict=dall)
                output_perturbation = np.concatenate((output_perturbation, perturb_pred_label))  # perturb results

            prop_same_label = np.mean(outputs_base == output_perturbation)  # proportion of same prediections
            results_robustness[layer, noise_id] = prop_same_label

            sys.stdout.flush()

        sys.stdout.flush()

    return results_robustness  # indexed [layer][noise_id]


def run(opt):

    ################################################################################################
    # Read experiment to run
    ################################################################################################

    # Skip execution if instructed in experiment
    if opt.skip:
        print("SKIP")
        quit()

    print('name:', opt.name)
    print('factor:', opt.dnn.factor)
    print('batch size:', opt.hyper.batch_size)

    ################################################################################################
    # get robustness
    ################################################################################################

    range_len = 7
    layers = opt.dnn.layers
    knockout_idx = [1, 2, 4]
    noise_idx = [0, 3]
    knockout_range = np.linspace(0.0, 1.0, num=range_len, endpoint=True)
    noise_range = np.linspace(0.0, 1.0, num=range_len, endpoint=True)

    results = np.zeros((NUM_PTYPES, layers+1, range_len))  # the +1 on layers is for all layers

    for ptype in range(NUM_PTYPES):
        # perturbation types: 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko
        if ptype in [0, 1]:  # for now, we skip over weight perturbations and only look at activation ones
            continue

        print("Perturbation type: " + str(ptype))

        if ptype in knockout_idx:
            range_robustness = knockout_range
        else:
            range_robustness = noise_range

        acc_1, acc_5, pred_label, label, handle, perturbation_params, select, val_iterator, num_iter_val \
            = create_graph(opt, ptype)
        # note that HERE ^ perturbation_params is indexed [type][layer] select is indexed [layer][neuron]

        # allow for GPU memory to be allocated as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            print("RESTORE")
            print(opt.log_dir_base + opt.name)

            if opt.dnn.name == 'resnet':
                saver = tf.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/'))

            elif opt.dnn.name == 'inception':
                variable_averages = tf.train.ExponentialMovingAverage(0.9999)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                # saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/'))

            ''' 
            ckpt = tf.train.get_checkpoint_state(opt.log_dir_base + opt.name + '/')
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    # Restores from checkpoint with absolute path.
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    # Restores from checkpoint with relative path.
                    saver.restore(sess, os.path.join(opt.log_dir_base + opt.name + '/',
                                                     ckpt.model_checkpoint_path))
            '''

            val_handle_full = sess.run(val_iterator.string_handle())

            # resuls indexed [pytpe, layer, parturbation_amount]
            results[ptype, :, :] = test_robustness(sess, pred_label, handle, perturbation_params,
                                                   select, opt, range_robustness, val_iterator,
                                                   val_handle_full, num_iter_val, ptype)

    with open(opt.results_dir + opt.name + '/robustness.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)

    tf.reset_default_graph()
    gc.collect()
    sys.stdout.flush()

    print(":)")
