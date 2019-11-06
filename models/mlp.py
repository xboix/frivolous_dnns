import tensorflow as tf
from utils import summary as summ
from numpy import *
import perturbations as pt


def MLP3(x, opt, labels_id, dropout_rate):
    parameters = []

    aa = x
    num_neurons_before_fc = int(prod(aa.get_shape()[1:]))
    flattened = tf.reshape(aa, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(tf.truncated_normal([num_neurons_before_fc, int(12 * opt.dnn.neuron_multiplier[0])],
                                            dtype=tf.float32, stddev=1e-3), name='weights')
        b = tf.Variable(0.1 * tf.ones([int(512 * opt.dnn.neuron_multiplier[0])]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope))
        summ.activation_summaries(fc1, opt)
        dropout1 = tf.nn.dropout(fc1, dropout_rate)

    # fc2
    with tf.name_scope('fc2') as scope:
        W = tf.Variable(
            tf.truncated_normal([int(512 * opt.dnn.neuron_multiplier[0]), int(512 * opt.dnn.neuron_multiplier[1])],
                                dtype=tf.float32, stddev=1e-3), name='weights')
        b = tf.Variable(0.1 * tf.ones([int(512 * opt.dnn.neuron_multiplier[1])]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dropout1, W), b, name=scope))
        summ.activation_summaries(fc2, opt)
        dropout2 = tf.nn.dropout(fc2, dropout_rate)

    # fc3
    with tf.name_scope('fc3') as scope:
        W = tf.Variable(
            tf.truncated_normal([int(512 * opt.dnn.neuron_multiplier[1]), int(512 * opt.dnn.neuron_multiplier[2])],
                                dtype=tf.float32, stddev=1e-3), name='weights')
        b = tf.Variable(0.1 * tf.ones([int(512 * opt.dnn.neuron_multiplier[2])]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dropout2, W), b, name=scope))
        summ.activation_summaries(fc3, opt)
        dropout3 = tf.nn.dropout(fc3, dropout_rate)

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(512 * opt.dnn.neuron_multiplier[2]), len(labels_id)],
                                            dtype=tf.float32,
                                            stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([len(labels_id)]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc8 = tf.nn.bias_add(tf.matmul(dropout3, W), b, name=scope)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters


def MLP1(x, opt):

    num_neurons_before_fc = int(x.get_shape()[1])
    activations = []

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable( tf.truncated_normal([num_neurons_before_fc, int(32*opt.dnn.neuron_multiplier[0])], dtype=tf.float32,
                                stddev=1e-3*opt.hyper.init_factor), name='weights')
        b = tf.Variable(0 * tf.ones([int(32*opt.dnn.neuron_multiplier[0])]), name='bias')

        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b, name=scope))
        activations += [fc1]
    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32*opt.dnn.neuron_multiplier[0]), 2],
                                            dtype=tf.float32, stddev=1e-2*opt.hyper.init_factor), name='weights')
        b = tf.Variable(tf.zeros([2]), name='bias')

        fc8 = tf.nn.bias_add(tf.matmul(fc1, W), b, name=scope)
        activations += [fc8]

    return fc8, [], activations


def MLP1_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type):
    # x is an input, opt is the experiment
    # select gives selected neurons where 1 indicates being selected, indexed as [layer][node]
    # labels_id=categories
    # dropout rate=dropout rate
    # perturbation_params=array of keep/drop probs indexed [type][layer]
    # perturbation_type is an int in range(5) giving perturbation type:
    # 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko
    parameters = []

    num_neurons_before_fc = int(x.get_shape()[1])
    flattened = tf.reshape(x, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(
            tf.truncated_normal([num_neurons_before_fc, int(32 * opt.dnn.neuron_multiplier[0])], dtype=tf.float32,
                                stddev=1e-3),
            name='weights')
        b = tf.Variable(0.1 * tf.ones([int(32 * opt.dnn.neuron_multiplier[0])]), name='bias')

        # weight perturbation
        if perturbation_type == 0:
            W = pt.weight_knockout(W, perturbation_params[1][0])
        if perturbation_type == 1:
            W = pt.weight_noise(W, perturbation_params[0][0])

        parameters += [W, b]

        fc1_predrop = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope))

        # if perturbation_type == 2:
        #     fc3_predrop = pt.activation_knockout(fc3_predrop, perturbation_params[2][2])
        if perturbation_type == 3:
            fc1_predrop = pt.activation_noise(fc1_predrop, perturbation_params[3][0], opt.hyper.batch_size)
        elif perturbation_type in [2, 4]:
            ss = tf.reshape(tf.tile(select[0], [opt.hyper.batch_size]), [-1, int(fc1_predrop.get_shape()[1])])
            fc1_predrop = pt.activation_knockout_mask(fc1_predrop, perturbation_params[4][0], ss)

        dropout1 = tf.nn.dropout(fc1_predrop, dropout_rate)

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32 * opt.dnn.neuron_multiplier[0]), len(labels_id)],
                                            dtype=tf.float32, stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([len(labels_id)]), name='bias')

        # weight perturbation
        if perturbation_type == 0:
            W = pt.weight_knockout(W, perturbation_params[1][1])
        if perturbation_type == 1:
            W = pt.weight_noise(W, perturbation_params[0][1])

        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc8 = tf.nn.bias_add(tf.matmul(dropout1, W), b, name=scope)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters


def MLP1_regression(x, opt):

    num_neurons_before_fc = int(x.get_shape()[1])
    activations = []

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(tf.truncated_normal([num_neurons_before_fc, int(32*opt.dnn.neuron_multiplier[0])],
                                            dtype=tf.float32, stddev=1e-3*opt.hyper.init_factor), name='weights')
        b = tf.Variable(0 * tf.ones([int(32*opt.dnn.neuron_multiplier[0])]), name='bias')

        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b, name=scope))
        activations += [fc1]
    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32*opt.dnn.neuron_multiplier[0]), 1],
                                            dtype=tf.float32, stddev=1e-2*opt.hyper.init_factor), name='weights')
        b = tf.Variable(tf.zeros([1]), name='bias')

        fc8 = tf.nn.bias_add(tf.matmul(fc1, W), b, name=scope)
        activations += [fc8]

    return fc8, [], activations


def MLP1_regression_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type):
    # x is an input, opt is the experiment
    # select gives selected neurons where 1 indicates being selected, indexed as [layer][node]
    # labels_id=categories
    # dropout rate=dropout rate
    # perturbation_params=array of keep/drop probs indexed [type][layer]
    # perturbation_type is an int in range(5) giving perturbation type:
    # 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko
    parameters = []

    num_neurons_before_fc = int(x.get_shape()[1])
    flattened = tf.reshape(x, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(
            tf.truncated_normal([num_neurons_before_fc, int(32 * opt.dnn.neuron_multiplier[0])], dtype=tf.float32,
                                stddev=1e-3),
            name='weights')
        b = tf.Variable(0.1 * tf.ones([int(32 * opt.dnn.neuron_multiplier[0])]), name='bias')

        # weight perturbation
        if perturbation_type == 0:
            W = pt.weight_knockout(W, perturbation_params[1][0])
        if perturbation_type == 1:
            W = pt.weight_noise(W, perturbation_params[0][0])

        parameters += [W, b]

        fc1_predrop = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope))

        # if perturbation_type == 2:
        #     fc3_predrop = pt.activation_knockout(fc3_predrop, perturbation_params[2][2])
        if perturbation_type == 3:
            fc1_predrop = pt.activation_noise(fc1_predrop, perturbation_params[3][0], opt.hyper.batch_size)
        elif perturbation_type in [2, 4]:
            ss = tf.reshape(tf.tile(select[0], [opt.hyper.batch_size]), [-1, int(fc1_predrop.get_shape()[1])])
            fc1_predrop = pt.activation_knockout_mask(fc1_predrop, perturbation_params[4][0], ss)

        dropout1 = tf.nn.dropout(fc1_predrop, dropout_rate)

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32 * opt.dnn.neuron_multiplier[0]), 1],
                                            dtype=tf.float32, stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([1]), name='bias')

        # weight perturbation
        if perturbation_type == 0:
            W = pt.weight_knockout(W, perturbation_params[1][1])
        if perturbation_type == 1:
            W = pt.weight_noise(W, perturbation_params[0][1])

        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc8 = tf.nn.bias_add(tf.matmul(dropout1, W), b, name=scope)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters


def MLP1_linear(x, opt):

    num_neurons_before_fc = int(x.get_shape()[1])
    activations = []

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(tf.truncated_normal([num_neurons_before_fc, int(32*opt.dnn.neuron_multiplier[0])], dtype=tf.float32,
                                stddev=1e-3*opt.hyper.init_factor), name='weights')
        b = tf.Variable(0 * tf.ones([int(32)*opt.dnn.neuron_multiplier[0]]), name='bias')

        fc1 = tf.nn.bias_add(tf.matmul(x, W), b, name=scope)
        activations += [fc1]

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32*opt.dnn.neuron_multiplier[0]), 2],
                                            dtype=tf.float32, stddev=1e-2*opt.hyper.init_factor), name='weights')
        b = tf.Variable(tf.zeros([2]), name='bias')

        fc8 = tf.nn.bias_add(tf.matmul(fc1, W), b, name=scope)
        activations += [fc8]
    return fc8, [], activations


def MLP1_linear_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type):
    # x is an input, opt is the experiment
    # select gives selected neurons where 1 indicates being selected, indexed as [layer][node]
    # labels_id=categories
    # dropout rate=dropout rate
    # perturbation_params=array of keep/drop probs indexed [type][layer]
    # perturbation_type is an int in range(5) giving perturbation type:
    # 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko
    parameters = []
    num_neurons_before_fc = int(x.get_shape()[1])
    flattened = tf.reshape(x, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(
            tf.truncated_normal([num_neurons_before_fc, int(32 * opt.dnn.neuron_multiplier[0])], dtype=tf.float32,
                                stddev=1e-3),
            name='weights')
        b = tf.Variable(0.1 * tf.ones([int(32 * opt.dnn.neuron_multiplier[0])]), name='bias')

        # weight perturbation
        if perturbation_type == 0:
            W = pt.weight_knockout(W, perturbation_params[1][0])
        if perturbation_type == 1:
            W = pt.weight_noise(W, perturbation_params[0][0])

        parameters += [W, b]

        fc1_predrop = tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope)

        # if perturbation_type == 2:
        #     fc3_predrop = pt.activation_knockout(fc3_predrop, perturbation_params[2][2])
        if perturbation_type == 3:
            fc1_predrop = pt.activation_noise(fc1_predrop, perturbation_params[3][0], opt.hyper.batch_size)
        elif perturbation_type in [2, 4]:
            ss = tf.reshape(tf.tile(select[0], [opt.hyper.batch_size]), [-1, int(fc1_predrop.get_shape()[1])])
            fc1_predrop = pt.activation_knockout_mask(fc1_predrop, perturbation_params[4][0], ss)

        dropout1 = tf.nn.dropout(fc1_predrop, dropout_rate)

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32 * opt.dnn.neuron_multiplier[0]), len(labels_id)],
                                            dtype=tf.float32, stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([len(labels_id)]), name='bias')

        # weight perturbation
        if perturbation_type == 0:
            W = pt.weight_knockout(W, perturbation_params[1][1])
        if perturbation_type == 1:
            W = pt.weight_noise(W, perturbation_params[0][1])

        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc8 = tf.nn.bias_add(tf.matmul(dropout1, W), b, name=scope)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters


def MLP1_linear_regression(x, opt):

    num_neurons_before_fc = int(x.get_shape()[1])
    activations = []

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(tf.truncated_normal([num_neurons_before_fc, int(32*opt.dnn.neuron_multiplier[0])],
                                            dtype=tf.float32,
                                            stddev=1e-3*opt.hyper.init_factor), name='weights')
        b = tf.Variable(0 * tf.ones([int(32)*opt.dnn.neuron_multiplier[0]]), name='bias')

        fc1 = tf.nn.bias_add(tf.matmul(x, W), b, name=scope)
        activations += [fc1]

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32*opt.dnn.neuron_multiplier[0]), 1],
                                            dtype=tf.float32, stddev=1e-2*opt.hyper.init_factor), name='weights')
        b = tf.Variable(tf.zeros([1]), name='bias')

        fc8 = tf.nn.bias_add(tf.matmul(fc1, W), b, name=scope)
        activations += [fc8]
    return fc8, [], activations


def MLP1_linear_regression_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type):
    # x is an input, opt is the experiment
    # select gives selected neurons where 1 indicates being selected, indexed as [layer][node]
    # labels_id=categories
    # dropout rate=dropout rate
    # perturbation_params=array of keep/drop probs indexed [type][layer]
    # perturbation_type is an int in range(5) giving perturbation type:
    # 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko
    parameters = []
    num_neurons_before_fc = int(x.get_shape()[1])
    flattened = tf.reshape(x, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(
            tf.truncated_normal([num_neurons_before_fc, int(32 * opt.dnn.neuron_multiplier[0])], dtype=tf.float32,
                                stddev=1e-3),
            name='weights')
        b = tf.Variable(0.1 * tf.ones([int(32 * opt.dnn.neuron_multiplier[0])]), name='bias')

        # weight perturbation
        if perturbation_type == 0:
            W = pt.weight_knockout(W, perturbation_params[1][0])
        if perturbation_type == 1:
            W = pt.weight_noise(W, perturbation_params[0][0])

        parameters += [W, b]

        fc1_predrop = tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope)

        # if perturbation_type == 2:
        #     fc3_predrop = pt.activation_knockout(fc3_predrop, perturbation_params[2][2])
        if perturbation_type == 3:
            fc1_predrop = pt.activation_noise(fc1_predrop, perturbation_params[3][0], opt.hyper.batch_size)
        elif perturbation_type in [2, 4]:
            ss = tf.reshape(tf.tile(select[0], [opt.hyper.batch_size]), [-1, int(fc1_predrop.get_shape()[1])])
            fc1_predrop = pt.activation_knockout_mask(fc1_predrop, perturbation_params[4][0], ss)

        dropout1 = tf.nn.dropout(fc1_predrop, dropout_rate)

    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(32 * opt.dnn.neuron_multiplier[0]), 1],
                                            dtype=tf.float32, stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([1]), name='bias')

        # weight perturbation
        if perturbation_type == 0:
            W = pt.weight_knockout(W, perturbation_params[1][1])
        if perturbation_type == 1:
            W = pt.weight_noise(W, perturbation_params[0][1])

        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc8 = tf.nn.bias_add(tf.matmul(dropout1, W), b, name=scope)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters

