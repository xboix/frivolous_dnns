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


def get_activations(opt, cross, max_samples=5e4):

    np.random.seed(cross)

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

    # Initialize dataset and creates TF records if they do not exist
    dataset = data.ImagenetDataset(opt)

    # Repeatable datasets for training
    val_dataset = dataset.create_dataset(set_name='val', repeat=True)

    # Handles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, val_dataset.output_types, val_dataset.output_shapes)

    val_iterator = val_dataset.make_initializable_iterator()
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

                to_call = getattr(nets, opt.dnn.name)

                logit, activation = to_call(_images, opt)

                tf.get_variable_scope().reuse_variables()
                logits_list.append(logit)
                activations_list.append(activation)

    logits = tf.reshape(tf.stack(logits_list, axis=0), [-1, 1001])

    activations = []
    num_gpus = len(activations_list)

    for layer in range(len(activations_list[0])):
        activations.append(tf.concat([activations_list[i][layer] for i in range(num_gpus)], axis=0))

    pred_label = tf.argmax(logits, axis=1)
    acc_1 = tf.nn.in_top_k(predictions=logits, targets=label, k=1, name='top_1_op')
    acc_5 = tf.nn.in_top_k(predictions=logits, targets=label, k=5, name='top_5_op')
    ################################################################################################

    config = tf.ConfigProto(allow_soft_placement=True)  # inter_op_parallelism_threads=80,
                            #intra_op_parallelism_threads=80,
    #config.gpu_options.allow_growth = True
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

        # Run one pass over a batch of the validation dataset.
        sess.run(val_iterator.initializer)

        acc_tmp_1 = 0.0
        acc_tmp_5 = 0.0
        total = 0
        activations_out = []
        labels_out = []

        total_iter = int(dataset.num_total_images / opt.hyper.batch_size) + 1

        max_samples_per_iter = int(max_samples / total_iter)

        for num_iter in range(total_iter):
            # note that we need activations_tmp to have shape [batch_size, height, width, channels]
            acc_val_1, acc_val_5, activations_tmp, labels_batch = sess.run(
                [acc_1, acc_5, activations, label],
                feed_dict={handle: val_handle_full})

            labels_tmp = []
            for layer in range(len(activations_tmp)):
                labels_tmp.append(np.repeat(labels_batch, np.prod(np.shape(activations_tmp[layer])[1:-1])))
                activations_tmp[layer] = np.reshape(activations_tmp[layer],
                                                    (-1, np.shape(activations_tmp[layer])[-1]))
                num_samples = np.shape(activations_tmp[layer])[0]
                if num_samples > max_samples_per_iter:
                    idx = np.random.permutation(num_samples)[: max_samples_per_iter]
                    activations_tmp[layer] = activations_tmp[layer][idx, :]
                    labels_tmp[layer] = labels_tmp[layer][idx]
            if not activations_out:
                activations_out = activations_tmp
                labels_out = labels_tmp
            else:
                for layer in range(len(activations_tmp)):
                    activations_out[layer] = np.append(activations_out[layer], activations_tmp[layer], axis=0)
                    labels_out[layer] = np.append(labels_out[layer], labels_tmp[layer], axis=0)

            for i in range(len(acc_val_1)):
                total += 1
                acc_tmp_1 += acc_val_1[i]
                acc_tmp_5 += acc_val_5[i]

            print('iteration:', str(num_iter) + '/' + str(total_iter-1), 'running_top-1:',
                  acc_tmp_1 / float(total), 'running_top-5:', acc_tmp_5 / float(total))
            sys.stdout.flush()

        if total > 0:
            ret_acc = acc_tmp_1 / float(total)
            ret_acc_5 = acc_tmp_5 / float(total)

        sys.stdout.flush()
        sess.close()

    return ret_acc, ret_acc_5, activations_out, labels_out


def run(opt):

    # Skip execution if instructed in experiment
    if opt.skip:
        print("SKIP")
        quit()

    print('name:', opt.name)
    print('factor:', opt.dnn.factor)
    print('batch size:', opt.hyper.batch_size)

    # make directory to put results into if not already exists
    if not os.path.exists(opt.results_dir + opt.name + '/'):
        os.mkdir(opt.results_dir + opt.name + '/')

    for cross in range(3):

        print('cross', cross)

        record_acc, record_acc_5, activations, gt = get_activations(opt, cross)

        with open(opt.results_dir + opt.name + '/activations' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(activations, f, protocol=2)
        with open(opt.results_dir + opt.name + '/labels' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(gt, f, protocol=2)
        with open(opt.results_dir + opt.name + '/accuracy' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump([record_acc, record_acc_5], f, protocol=2)

        tf.reset_default_graph()
        gc.collect()
        print("top-1:", record_acc)
        print("top-5: ", record_acc_5)
        sys.stdout.flush()

    print(":)")
