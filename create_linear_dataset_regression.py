import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

################################################################################################
# Declare DNN
################################################################################################


def MLP1_regression(x, dim):
    aa = x
    num_neurons_before_fc = dim
    flattened = tf.reshape(aa, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(tf.truncated_normal([num_neurons_before_fc, int(32)], dtype=tf.float32,
                                            stddev=1e-3), name='weights')
        b = tf.Variable(0 * tf.ones([int(32)]), name='bias')

        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope))

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32), 1], dtype=tf.float32, stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([1]), name='bias')

        fc8 = tf.nn.bias_add(tf.matmul(fc1, W), b, name=scope)

    return fc8


for dim in [10, 10000]:

    tf.reset_default_graph()

    batch_size = 1000
    input = tf.placeholder(tf.float32, shape=(batch_size, dim))

    y = MLP1_regression(input, dim)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for set in ['train', 'val', 'test']:
            data = []
            labels = []
            for i in range(50):
                xx = np.reshape(np.random.normal(0, 1, batch_size*dim), [batch_size, dim])
                data.append(xx)
                a = sess.run(y, feed_dict={input: xx})
                labels.append(np.squeeze(a))

            data = np.concatenate(data)
            labels = np.concatenate(labels)

            dataset = {'data': data, 'labels': labels}
            import pickle

            with open(set + '_' + str(dim) + '.pickle', 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


