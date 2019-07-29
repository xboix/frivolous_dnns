import tensorflow as tf
from utils import summary as summ
import perturbations as pt
from numpy import *
from keras import backend as K
import numpy as np
import sys
num_neurons = [96, 256, 384, 192]


def Alexnet(x, opt, labels_id, dropout_rate):

    reuse = False
    global num_neurons

    parameters = []
    activations = []

    init_type = tf.glorot_normal_initializer
    if opt.init_type == 1:
        init_type = tf.glorot_uniform_initializer
    elif opt.init_type == 2:
        init_type = tf.keras.initializers.he_normal
    elif opt.init_type == 3:
        init_type = tf.keras.initializers.he_uniform
    elif opt.init_type == 4:
        init_type = tf.keras.initializers.lecun_normal
    elif opt.init_type == 5:
        init_type = tf.keras.initializers.lecun_uniform

    f_act = tf.nn.relu
    if opt.act_function == 1:
        f_act = tf.nn.leaky_relu
    elif opt.act_function == 2:
        f_act = tf.nn.elu
    elif opt.act_function == 3:
        f_act = tf.nn.selu
    elif opt.act_function == 4:
        f_act = tf.nn.sigmoid
    elif opt.act_function == 5:
        f_act = tf.nn.tanh

    # conv1
    with tf.variable_scope('conv1', reuse=reuse) as scope:
        kernel = tf.get_variable(shape=[5, 5, 3, int(num_neurons[0] * opt.dnn.neuron_multiplier[0])],
                                 initializer=init_type(), name='weights')
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(
            initializer=tf.constant(0.0, shape=[int(num_neurons[0] * opt.dnn.neuron_multiplier[0])]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = f_act(pre_activation, name=scope.name)

        print("Conv 1 cells = " + str(np.prod(conv1.shape[1:])))
        print("Conv 1 params = " + str(np.prod(kernel.shape)))

        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv1, opt)
        activations += [conv1]

    # pool1
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        summ.activation_summaries(pool1, opt)

    # lrn1
    with tf.variable_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse) as scope:
        kernel = tf.get_variable(shape=[5, 5, int(num_neurons[0] * opt.dnn.neuron_multiplier[0]),
                                        int(num_neurons[1] * opt.dnn.neuron_multiplier[1])],
                                 initializer=init_type(), name='weights')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(
            initializer=tf.constant(0.1, shape=[int(num_neurons[1] * opt.dnn.neuron_multiplier[1])]), name='biases')

        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = f_act(pre_activation, name=scope.name)

        print("Conv 2 cells = " + str(np.prod(conv2.shape[1:])))
        print("Conv 2 params = " + str(np.prod(kernel.shape)))

        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv2, opt)
        activations += [conv2]

    # pool2
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        summ.activation_summaries(pool2, opt)

    # lrn2
    with tf.variable_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.)

    # fc3 (fc6 in Alexnet)
    with tf.variable_scope('fc3', reuse=reuse) as scope:

        dim1 = int(np.prod(lrn2.get_shape()[1:]))
        dim2 = int(num_neurons[2] * opt.dnn.neuron_multiplier[2])
        flattened = tf.reshape(lrn2, [opt.hyper.batch_size, -1])

        weights = tf.get_variable(shape=[dim1, dim2], initializer=init_type(), name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.1, shape=[dim2]), name='biases')

        fc3_predrop = f_act(tf.matmul(flattened, weights) + biases, name=scope.name)
        fc3 = tf.nn.dropout(fc3_predrop, dropout_rate)

        print("FC 1 cells = " + str(np.prod(fc3.shape[1:])))
        print("FC 1 params = " + str(np.prod(weights.shape)))

        activations += [fc3]
        parameters += [weights]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(fc3, opt)

    # fc4 (fc7 in Alexnet)
    with tf.variable_scope('fc4', reuse=reuse) as scope:
        dim3 = int(num_neurons[3] * opt.dnn.neuron_multiplier[3])
        weights = tf.get_variable(shape=[dim2, dim3], initializer=init_type(), name='weights')
        biases = tf.get_variable(
            initializer=tf.constant(0.1, shape=[int(num_neurons[3] * opt.dnn.neuron_multiplier[3])]), name='biases')
        fc4_predrop = f_act(tf.matmul(fc3, weights) + biases, name=scope.name)
        fc4 = tf.nn.dropout(fc4_predrop, dropout_rate)

        print("FC 2 cells = " + str(np.prod(fc4.shape[1:])))
        print("FC 2 params = " + str(np.prod(weights.shape)))

        activations += [fc4]
        parameters += [weights]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(fc4, opt)

    # linear softmax (fc8 in Alexnet)
    # We don't apply softmax--tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency
    with tf.variable_scope('softmax_linear', reuse=reuse) as scope:
        weights = tf.get_variable(shape=[dim3, len(labels_id)], initializer=init_type(), name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[len(labels_id)]), name='biases')
        fc5 = tf.add(tf.matmul(fc4, weights), biases, name=scope.name)

        print("Out params = " + str(np.prod(weights.shape)))

        activations += [fc5]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(fc5, opt)

    return fc5, parameters, activations


def Alexnet_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type):
    # x is an input, opt is the experiment
    # select gives selected neurons where 1 indicates being selected, indexed as [layer][node]
    # labels_id=categories
    # dropout rate=dropout rate
    # perturbation_params=array of keep/drop probs indexed [type][layer]
    # perturbation_type is an int in range(5) giving perturbation type:
    # 0=weight noise, 1=weight ko, 2=act ko, 3=act noise, 4=targeted act ko
    
    reuse = False
    global num_neurons
    
    parameters = []

    init_type = tf.glorot_normal_initializer
    if opt.init_type == 1:
        init_type = tf.glorot_uniform_initializer
    elif opt.init_type == 2:
        init_type = tf.keras.initializers.he_normal
    elif opt.init_type == 3:
        init_type = tf.keras.initializers.he_uniform
    elif opt.init_type == 4:
        init_type = tf.keras.initializers.lecun_normal
    elif opt.init_type == 5:
        init_type = tf.keras.initializers.lecun_uniform

    f_act = tf.nn.relu
    if opt.act_function == 1:
        f_act = tf.nn.leaky_relu
    elif opt.act_function == 2:
        f_act = tf.nn.elu
    elif opt.act_function == 3:
        f_act = tf.nn.selu
    elif opt.act_function == 4:
        f_act = tf.nn.sigmoid
    elif opt.act_function == 5:
        f_act = tf.nn.tanh
        
    with tf.variable_scope('conv1', reuse=reuse) as scope:
        kernel = tf.get_variable(shape=[5, 5, 3, int(num_neurons[0] * opt.dnn.neuron_multiplier[0])],
                                 initializer=init_type(), name='weights')
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(
            initializer=tf.constant(0.0, shape=[int(num_neurons[0] * opt.dnn.neuron_multiplier[0])]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = f_act(pre_activation, name=scope.name)

        # Activation perturbation
        # if perturbation_type == 2:
        #     conv1 = pt.activation_knockout(conv1, perturbation_params[2][0])
        if perturbation_type == 3:
            conv1 = pt.activation_noise(conv1, perturbation_params[3][0], opt.hyper.batch_size)
        elif perturbation_type in [2, 4]:
            ss = tf.reshape(tf.tile(select[0], [int(np.prod(conv1.get_shape()[1:3])) * opt.hyper.batch_size]),
                            [-1, int(conv1.get_shape()[1]), int(conv1.get_shape()[2]), int(conv1.get_shape()[3])])
            conv1 = pt.activation_knockout_mask(conv1, perturbation_params[4][0], ss)  # ss is the mask

        parameters += [kernel, biases]
        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv1, opt)

    # pool1
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        summ.activation_summaries(pool1, opt)

    # lrn1
    with tf.variable_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse) as scope:
        kernel = tf.get_variable(shape=[5, 5, int(num_neurons[0] * opt.dnn.neuron_multiplier[0]),
                                        int(num_neurons[1] * opt.dnn.neuron_multiplier[1])],
                                 initializer=init_type(), name='weights')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(
            initializer=tf.constant(0.1, shape=[int(num_neurons[1] * opt.dnn.neuron_multiplier[1])]), name='biases')

        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = f_act(pre_activation, name=scope.name)

        # Activation perturbation
        # if perturbation_type == 2:
        #     conv2 = pt.activation_knockout(conv2, perturbation_params[2][1])
        if perturbation_type == 3:
            conv2 = pt.activation_noise(conv2, perturbation_params[3][1], opt.hyper.batch_size)
        elif perturbation_type in [2, 4]:
            ss = tf.reshape(tf.tile(select[1], [int(np.prod(conv2.get_shape()[1:3])) * opt.hyper.batch_size]),
                            [-1, int(conv2.get_shape()[1]), int(conv2.get_shape()[2]), int(conv2.get_shape()[3])])
            conv2 = pt.activation_knockout_mask(conv2, perturbation_params[4][1], ss)

        parameters += [kernel, biases]
        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv2, opt)

    # pool2
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        summ.activation_summaries(pool2, opt)

    # lrn2
    with tf.variable_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.)

    # fc3 (6 in Alexnet)
    with tf.variable_scope('fc3', reuse=reuse) as scope:

        dim1 = int(np.prod(lrn2.get_shape()[1:]))
        dim2 = int(num_neurons[2] * opt.dnn.neuron_multiplier[2])
        flattened = tf.reshape(lrn2, [opt.hyper.batch_size, -1])

        weights = tf.get_variable(shape=[dim1, dim2], initializer=init_type(), name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.1, shape=[dim2]), name='biases')

        # weight perturbation
        if perturbation_type == 0:
            weights = pt.weight_knockout(weights, perturbation_params[1][2])
        if perturbation_type == 1:
            weights = pt.weight_noise(weights, perturbation_params[0][2])

        parameters += [weights, biases]
        summ.variable_summaries(weights, biases, opt)

        fc3_predrop = f_act(tf.matmul(flattened, weights) + biases, name=scope.name)

        # activation perturbation
        # if perturbation_type == 2:
        #     fc3_predrop = pt.activation_knockout(fc3_predrop, perturbation_params[2][2])
        if perturbation_type == 3:
            fc3_predrop = pt.activation_noise(fc3_predrop, perturbation_params[3][2], opt.hyper.batch_size)
        elif perturbation_type in [2, 4]:
            ss = tf.reshape(tf.tile(select[2], [opt.hyper.batch_size]), [-1, int(fc3_predrop.get_shape()[1])])
            fc3_predrop = pt.activation_knockout_mask(fc3_predrop, perturbation_params[4][2], ss)

        fc3 = tf.nn.dropout(fc3_predrop, dropout_rate)
        summ.activation_summaries(fc3, opt)

    # fc4 (fc7 in Alexnet)
    with tf.variable_scope('fc4', reuse=reuse) as scope:
        dim3 = int(num_neurons[3] * opt.dnn.neuron_multiplier[3])
        weights = tf.get_variable(shape=[dim2, dim3], initializer=init_type(), name='weights')
        biases = tf.get_variable(
            initializer=tf.constant(0.1, shape=[int(num_neurons[3] * opt.dnn.neuron_multiplier[3])]), name='biases')

        # weight perturbation
        if perturbation_type == 0:
            weights = pt.weight_knockout(weights, perturbation_params[1][3])
        if perturbation_type == 1:
            weights = pt.weight_noise(weights, perturbation_params[0][3])

        parameters += [weights, biases]
        summ.variable_summaries(weights, biases, opt)

        fc4_predrop = f_act(tf.matmul(fc3, weights) + biases, name=scope.name)

        # activation perturbation
        # if perturbation_type == 2:
        #     fc4_predrop = pt.activation_knockout(fc4_predrop, perturbation_params[2][3])
        if perturbation_type == 3:
            fc4_predrop = pt.activation_noise(fc4_predrop, perturbation_params[3][3], opt.hyper.batch_size)
        elif perturbation_type in [2, 4]:
            ss = tf.reshape(tf.tile(select[3], [opt.hyper.batch_size]), [-1, int(fc4_predrop.get_shape()[1])])
            fc4_predrop = pt.activation_knockout_mask(fc4_predrop, perturbation_params[4][3], ss)

        fc4 = tf.nn.dropout(fc4_predrop, dropout_rate)
        summ.activation_summaries(fc4, opt)

    # linear softmax (fc8 in Alexnet)
    # We don't apply softmax--tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency
    with tf.variable_scope('softmax_linear', reuse=reuse) as scope:
        weights = tf.get_variable(shape=[dim3, len(labels_id)], initializer=init_type(), name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[len(labels_id)]), name='biases')
        fc5 = tf.add(tf.matmul(fc4, weights), biases, name=scope.name)

        # weight perturbation
        if perturbation_type == 0:
            weights = pt.weight_knockout(weights, perturbation_params[1][4])
        if perturbation_type == 1:
            weights = pt.weight_noise(weights, perturbation_params[0][4])

        parameters += [weights, biases]
        summ.variable_summaries(weights, biases, opt)

        fc5 = tf.add(tf.matmul(fc4, weights), biases, name=scope.name)
        summ.activation_summaries(fc5, opt)

    return fc5, parameters
