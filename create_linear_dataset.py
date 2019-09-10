import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

################################################################################################
# Read experiment to run
################################################################################################

DIMENSIONALITY = 10
################################################################################################

################################################################################################
# Declare DNN
################################################################################################

batch_size = 1000
input = tf.placeholder(tf.float32, shape=(batch_size, DIMENSIONALITY))


def MLP1(x):

    aa = x
    num_neurons_before_fc = DIMENSIONALITY
    flattened = tf.reshape(aa, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable( tf.truncated_normal([num_neurons_before_fc, int(32)], dtype=tf.float32,
                                stddev=1e-3), name='weights')
        b = tf.Variable(0 * tf.ones([int(32)]), name='bias')

        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope))

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32 ), 2],  dtype=tf.float32, stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([2]), name='bias')

        fc8 = tf.nn.bias_add(tf.matmul(fc1, W), b, name=scope)

    return fc8

y = MLP1(input)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for set in ['train', 'val', 'test']:
        data = []
        labels = []
        for i in range(100):
            xx = np.reshape(np.random.normal(0, 1, batch_size*DIMENSIONALITY), [batch_size, DIMENSIONALITY])
            data.append(xx)
            a = sess.run(y, feed_dict={input: xx})
            labels.append(np.argmax(a, axis=1))

        data = np.concatenate(data)
        labels = np.concatenate(labels)

        dataset = {'data': data, 'labels': labels}
        import pickle

        with open(set + '_' + str(DIMENSIONALITY) + '.pickle', 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


