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


def evaluate_network(opt):
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

                logit, activations = to_call(_images, opt)

                tf.get_variable_scope().reuse_variables()
                logits_list.append(logit)

    logits = tf.reshape(tf.stack(logits_list, axis=0), [-1, 1001])
    pred_label = tf.argmax(logits, axis=1)
    acc_1 = tf.nn.in_top_k(predictions=logits, targets=label, k=1, name='top_1_op')
    acc_5 = tf.nn.in_top_k(predictions=logits, targets=label, k=5, name='top_5_op')
    ################################################################################################

    config = tf.ConfigProto(allow_soft_placement=True)#inter_op_parallelism_threads=80,
                            #intra_op_parallelism_threads=80,
                            #
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

        total_iter = int(dataset.num_total_images / opt.hyper.batch_size) + 1
        for num_iter in range(total_iter):
            acc_val_1, acc_val_5 = sess.run(
                [acc_1, acc_5],
                feed_dict={handle: val_handle_full})

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

    return ret_acc, ret_acc_5


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
    # Define training and validation datasets through Dataset API
    ################################################################################################
    record_acc, record_acc_5 = evaluate_network(opt)

    tf.reset_default_graph()
    gc.collect()
    print("top-1:", record_acc)
    print("top-5: ", record_acc_5)
    sys.stdout.flush()

    print(":)")



