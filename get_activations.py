from __future__ import print_function
import sys
import numpy as np
import os
import tensorflow as tf
import experiments
from data import cifar_dataset
from models import nets
import pickle
import time
from numpy import linalg as LA

################################################################################################
# Read experiment to run
################################################################################################

ID = int(sys.argv[1:][0])

opt = experiments.opt[ID]

# Skip execution if instructed in experiment
if opt.skip:
    print("SKIP")
    quit()

print('Experiment:', opt.name)

tf.compat.v1.set_random_seed(opt.seed)

if opt.hyper.mse:
    metric_name = 'mse'
else:
    metric_name = 'accuracy'

################################################################################################

MAX_SAMPLES = 5e4


def create_graph():
    ################################################################################################
    # Define training and validation datasets thorugh Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist
    # Initialize dataset and creates TF records if they do not exist
    if opt.dataset.dataset_name == 'cifar':
        from data import cifar_dataset
        dataset = cifar_dataset.Cifar10(opt)
    elif opt.dataset.dataset_name == 'rand10':
        from data import rand10_dataset
        dataset = rand10_dataset.Rand10(opt)
    elif opt.dataset.dataset_name == 'rand10000':
        from data import rand10000_dataset
        dataset = rand10000_dataset.Rand10000(opt)
    elif opt.dataset.dataset_name == 'rand10_regression':
        from data import rand10_regression_dataset
        dataset = rand10_regression_dataset.Rand10_regression(opt)
    elif opt.dataset.dataset_name == 'rand10000_regression':
        from data import rand10000_regression_dataset
        dataset = rand10000_regression_dataset.Rand10000_regression(opt)

    # No repeatable dataset for testing
    train_dataset_full = dataset.create_dataset(augmentation=False, standarization=True,
                                                set_name='train', repeat=False)
    val_dataset_full = dataset.create_dataset(augmentation=False, standarization=True,
                                              set_name='val', repeat=False)
    test_dataset_full = dataset.create_dataset(augmentation=False, standarization=True,
                                               set_name='test', repeat=False)

    # Hadles to switch datasets
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle, val_dataset_full.output_types, val_dataset_full.output_shapes)

    train_iterator_full = train_dataset_full.make_initializable_iterator()
    val_iterator_full = val_dataset_full.make_initializable_iterator()
    test_iterator_full = test_dataset_full.make_initializable_iterator()
    ################################################################################################

    ################################################################################################
    # DNN
    ################################################################################################

    # Get data from dataset
    image, y_ = iterator.get_next()
    if opt.dataset.dataset_name == 'cifar':
        image = tf.image.resize_images(image, [opt.hyper.image_size, opt.hyper.image_size])
        if opt.extense_summary:
            tf.summary.image('input', image)
    elif opt.dataset.dataset_name == 'rand10' or opt.dataset.dataset_name == 'rand10_regression':
        image = tf.compat.v1.reshape(image, [-1, 10])
    elif opt.dataset.dataset_name == 'rand10000' or opt.dataset.dataset_name == 'rand10000_regression':
        image = tf.compat.v1.reshape(image, [-1, 10000])

    # Call DNN
    dropout_rate = tf.compat.v1.placeholder(tf.float32)
    to_call = getattr(nets, opt.dnn.name)
    y, parameters, activations = to_call(image, dropout_rate, opt, dataset.list_labels)

    if opt.hyper.mse:
        # MSE metric
        metric = tf.reduce_mean((y_ - y) ** 2)
    else:
        # Accuracy metric
        im_prediction = tf.equal(tf.argmax(y, 1), y_)
        im_prediction = tf.cast(im_prediction, tf.float32)
        metric = tf.reduce_mean(im_prediction)

    num_iter_train = int(dataset.num_images_training * opt.dataset.proportion_training_set
                         / opt.hyper.batch_size) - 1
    num_iter_test = int(dataset.num_images_test / opt.hyper.batch_size) - 1
    num_iter_val = int(dataset.num_images_val / opt.hyper.batch_size) - 1

    return activations, metric, y_, handle, dropout_rate, train_iterator_full, val_iterator_full, \
           test_iterator_full, num_iter_train, num_iter_val, num_iter_test
    ################################################################################################


def get_activations(handle, metric, gt, dropout_rate, activations, opt,
                    iterator_dataset, handle_dataset, num_iter, cross):
    sess.run(iterator_dataset.initializer)

    np.random.seed(cross)

    activations_out = []
    metric_tmp = 0.0

    max_samples_per_iter = int(MAX_SAMPLES / num_iter)

    for _ in range(num_iter):
        activations_tmp, metric_batch, labels_batch = \
            sess.run([activations, metric, gt], feed_dict={handle: handle_dataset, dropout_rate: opt.hyper.drop_test})

        print([l.shape for l in activations_tmp])
        quit()
        sys.stdout.exit()

        metric_tmp += metric_batch

        labels_tmp = []
        for layer in range(len(activations_tmp)):
            labels_tmp.append(np.repeat(labels_batch, np.prod(np.shape(activations_tmp[layer])[1:-1])))
            activations_tmp[layer] = np.reshape(activations_tmp[layer], (-1, np.shape(activations_tmp[layer])[-1]))
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

    metric_out = metric_tmp / num_iter
    return activations_out, metric_out, labels_out


t0 = time.time()
# if not os.path.isfile(opt.log_dir_base + opt.name + '/activations0.pkl'):
activations, metric, gt, handle, dropout_rate, train_iterator_full, val_iterator_full, test_iterator_full,\
    num_iter_train, num_iter_val, num_iter_test = create_graph()

# results = [[[] for i in range(2)] for i in range(6)]

with tf.Session() as sess:
    ################################################################################################
    # Set up checkpoints and data
    ################################################################################################

    print("RESTORE")
    print(opt.log_dir_base + opt.name)

    saver = tf.compat.v1.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)
    if os.path.isdir(opt.log_dir_base + opt.name + '/models/'):
        saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
    elif os.path.isdir(opt.log_dir_base + opt.name + '/'):
        saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/'))
    else:
        print('Can\'t find model save path.')
        quit()

    ################################################################################################

    ################################################################################################
    # RUN TEST
    ################################################################################################
    train_handle_full = sess.run(train_iterator_full.string_handle())
    validation_handle_full = sess.run(val_iterator_full.string_handle())
    test_handle_full = sess.run(test_iterator_full.string_handle())

    for cross in range(3):
        print('cross:', cross)
        res, met, gt_labels_train = get_activations(handle, metric, gt, dropout_rate, activations, opt,
                                                    train_iterator_full, train_handle_full, num_iter_train, cross)
        if cross == 0:
            print("Train", metric_name+':',  met)
        with open(opt.log_dir_base + opt.name + '/activations' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(res, f, protocol=2)
        with open(opt.log_dir_base + opt.name + '/labels' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(gt_labels_train, f, protocol=2)
        with open(opt.log_dir_base + opt.name + '/' + metric_name + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(met, f, protocol=2)

        res_test, met_test, gt_labels_test = get_activations(handle, metric, gt, dropout_rate, activations, opt,
                                                             test_iterator_full, test_handle_full, num_iter_test,
                                                             cross)
        if cross == 0:
            print("Test", metric_name+':', met_test)
        with open(opt.log_dir_base + opt.name + '/activations_test' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(res_test, f, protocol=2)
        with open(opt.log_dir_base + opt.name + '/labels_test' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(gt_labels_test, f, protocol=2)
        with open(opt.log_dir_base + opt.name + '/' + metric_name + '_test' + str(cross) + '.pkl', 'wb') as f:
            pickle.dump(met_test, f, protocol=2)

        sys.stdout.flush()


tf.reset_default_graph()
t1 = time.time()
sys.stdout.flush()
print('Time: ', t1 - t0)
print('get_activations.py')
print(':)')
