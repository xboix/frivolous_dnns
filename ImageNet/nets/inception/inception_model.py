# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inception-v3 expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from nets.inception import ops
from nets.inception import scopes
from nets import perturbations as pt


def inception_v3(inputs, opt,
                 factor=1,
                 factor_end=1,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=False,
                 restore_logits=True,
                 scope=None):
    """Latest Inception from http://arxiv.org/abs/1512.00567.

      "Rethinking the Inception Architecture for Computer Vision"

      Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
      Zbigniew Wojna

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      dropout_keep_prob: dropout keep_prob.
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      restore_logits: whether or not the logits layers should be restored.
        Useful for fine-tuning a model with different num_classes.
      scope: Optional scope for name_scope.

    Returns:
      a list containing 'logits', 'aux_logits' Tensors.
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }

    end_points = {}
    activations = []

    with scopes.arg_scope([ops.conv2d, ops.fc], weight_decay=0.00004):
        with scopes.arg_scope([ops.conv2d],
                              stddev=0.1,
                              activation=tf.nn.relu,
                              batch_norm_params=batch_norm_params):
            with tf.name_scope(scope, 'inception_v3', [inputs]):
                with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                                      is_training=is_training):
                    with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                          stride=1, padding='VALID'):
                        # 299 x 299 x 3
                        end_points['conv0'] = ops.conv2d(inputs, int(32 * factor), [3, 3], stride=2,
                                                         scope='conv0')
                        activations.append(end_points['conv0'])
                        # 149 x 149 x 32
                        end_points['conv1'] = ops.conv2d(end_points['conv0'], int(32 * factor), [3, 3],
                                                         scope='conv1')
                        activations.append(end_points['conv1'])
                        # 147 x 147 x 32
                        end_points['conv2'] = ops.conv2d(end_points['conv1'], int(64 * factor), [3, 3],
                                                         padding='SAME', scope='conv2')
                        activations.append(end_points['conv2'])
                        # 147 x 147 x 64
                        end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                                           stride=2, scope='pool1')
                        # 73 x 73 x 64
                        end_points['conv3'] = ops.conv2d(end_points['pool1'], int(80 * factor), [1, 1],
                                                         scope='conv3')
                        activations.append(end_points['conv3'])
                        # 73 x 73 x 80.
                        end_points['conv4'] = ops.conv2d(end_points['conv3'], int(192 * factor), [3, 3],
                                                         scope='conv4')
                        activations.append(end_points['conv4'])
                        # 71 x 71 x 192.
                        end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                                           stride=2, scope='pool2')
                        # 35 x 35 x 192.
                        net = end_points['pool2']

                    # Inception blocks
                    with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                          stride=1, padding='SAME'):
                        # mixed: 35 x 35 x 256.
                        with tf.variable_scope('mixed_35x35x256a'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(64 * factor), [1, 1])
                            with tf.variable_scope('branch5x5'):
                                branch5x5 = ops.conv2d(net, int(48 * factor), [1, 1])
                                branch5x5 = ops.conv2d(branch5x5, int(64 * factor), [5, 5])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(64 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(32 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                            end_points['mixed_35x35x256a'] = net
                            activations.append(net)
                        # mixed_1: 35 x 35 x 288.
                        with tf.variable_scope('mixed_35x35x288a'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(64 * factor), [1, 1])
                            with tf.variable_scope('branch5x5'):
                                branch5x5 = ops.conv2d(net, int(48 * factor), [1, 1])
                                branch5x5 = ops.conv2d(branch5x5, int(64 * factor), [5, 5])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(64 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(64 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                            end_points['mixed_35x35x288a'] = net
                            activations.append(net)
                        # mixed_2: 35 x 35 x 288.
                        with tf.variable_scope('mixed_35x35x288b'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(64 * factor * factor_end), [1, 1])
                            with tf.variable_scope('branch5x5'):
                                branch5x5 = ops.conv2d(net, int(48 * factor * factor_end), [1, 1])
                                branch5x5 = ops.conv2d(branch5x5, int(64 * factor * factor_end), [5, 5])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(64 * factor * factor_end), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor * factor_end), [3, 3])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor * factor_end), [3, 3])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(64 * factor * factor_end), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                            end_points['mixed_35x35x288b'] = net
                            activations.append(net)
                        # mixed_3: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768a'):
                            with tf.variable_scope('branch3x3'):
                                branch3x3 = ops.conv2d(net, int(384 * factor), [3, 3], stride=2, padding='VALID')
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(64 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3],
                                                          stride=2, padding='VALID')
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                            net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])
                            end_points['mixed_17x17x768a'] = net
                            activations.append(net)
                        # mixed4: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768b'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(192 * factor), [1, 1])
                            with tf.variable_scope('branch7x7'):
                                branch7x7 = ops.conv2d(net, int(128 * factor), [1, 1])
                                branch7x7 = ops.conv2d(branch7x7, int(128 * factor), [1, 7])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [7, 1])
                            with tf.variable_scope('branch7x7dbl'):
                                branch7x7dbl = ops.conv2d(net, int(128 * factor), [1, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(128 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(128 * factor), [1, 7])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(128 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                            end_points['mixed_17x17x768b'] = net
                            activations.append(net)
                        # mixed_5: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768c'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(192 * factor), [1, 1])
                            with tf.variable_scope('branch7x7'):
                                branch7x7 = ops.conv2d(net, int(160 * factor), [1, 1])
                                branch7x7 = ops.conv2d(branch7x7, int(160 * factor), [1, 7])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [7, 1])
                            with tf.variable_scope('branch7x7dbl'):
                                branch7x7dbl = ops.conv2d(net, int(160 * factor), [1, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [1, 7])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                            end_points['mixed_17x17x768c'] = net
                            activations.append(net)
                        # mixed_6: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768d'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(192 * factor), [1, 1])
                            with tf.variable_scope('branch7x7'):
                                branch7x7 = ops.conv2d(net, int(160 * factor), [1, 1])
                                branch7x7 = ops.conv2d(branch7x7, int(160 * factor), [1, 7])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [7, 1])
                            with tf.variable_scope('branch7x7dbl'):
                                branch7x7dbl = ops.conv2d(net, int(160 * factor), [1, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [1, 7])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                            end_points['mixed_17x17x768d'] = net
                            activations.append(net)
                        # mixed_7: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768e'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(192 * factor), [1, 1])
                            with tf.variable_scope('branch7x7'):
                                branch7x7 = ops.conv2d(net, int(192 * factor), [1, 1])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [1, 7])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [7, 1])
                            with tf.variable_scope('branch7x7dbl'):
                                branch7x7dbl = ops.conv2d(net, int(192 * factor), [1, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                            end_points['mixed_17x17x768e'] = net
                            activations.append(net)
                        # Auxiliary Head logits
                        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
                        with tf.variable_scope('aux_logits'):
                            aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
                                                      padding='VALID')
                            aux_logits = ops.conv2d(aux_logits, int(128 * factor), [1, 1], scope='proj')
                            # Shape of feature map before the final layer.
                            shape = aux_logits.get_shape()
                            aux_logits = ops.conv2d(aux_logits, int(768 * factor), shape[1:3], stddev=0.01,
                                                    padding='VALID')
                            aux_logits = ops.flatten(aux_logits)
                            aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                                                stddev=0.001, restore=restore_logits)
                            end_points['aux_logits'] = aux_logits
                        # mixed_8: 8 x 8 x 1280.
                        # Note that the scope below is not changed to not void previous
                        # checkpoints.
                        # (TODO) Fix the scope when appropriate.
                        with tf.variable_scope('mixed_17x17x1280a'):
                            with tf.variable_scope('branch3x3'):
                                branch3x3 = ops.conv2d(net, int(192 * factor), [1, 1])
                                branch3x3 = ops.conv2d(branch3x3, int(320 * factor), [3, 3], stride=2,
                                                       padding='VALID')
                            with tf.variable_scope('branch7x7x3'):
                                branch7x7x3 = ops.conv2d(net, int(192 * factor), [1, 1])
                                branch7x7x3 = ops.conv2d(branch7x7x3, int(192 * factor), [1, 7])
                                branch7x7x3 = ops.conv2d(branch7x7x3, int(192 * factor), [7, 1])
                                branch7x7x3 = ops.conv2d(branch7x7x3, int(192 * factor), [3, 3],
                                                         stride=2, padding='VALID')
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                            net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
                            end_points['mixed_17x17x1280a'] = net
                            activations.append(net)
                        # mixed_9: 8 x 8 x 2048.
                        with tf.variable_scope('mixed_8x8x2048a'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(320 * factor), [1, 1])
                            with tf.variable_scope('branch3x3'):
                                branch3x3 = ops.conv2d(net, int(384 * factor), [1, 1])
                                branch3x3 = tf.concat(axis=3, values=[ops.conv2d(branch3x3, int(384 * factor), [1, 3]),
                                                                      ops.conv2d(branch3x3, int(384 * factor), [3, 1])])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(448 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(384 * factor), [3, 3])
                                branch3x3dbl = tf.concat(axis=3,
                                                         values=[ops.conv2d(branch3x3dbl, int(384 * factor), [1, 3]),
                                                                 ops.conv2d(branch3x3dbl, int(384 * factor), [3, 1])])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
                            end_points['mixed_8x8x2048a'] = net
                            activations.append(net)
                        # mixed_10: 8 x 8 x 2048.
                        with tf.variable_scope('mixed_8x8x2048b'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(320 * factor), [1, 1])
                            with tf.variable_scope('branch3x3'):
                                branch3x3 = ops.conv2d(net, int(384 * factor), [1, 1])
                                branch3x3 = tf.concat(axis=3, values=[ops.conv2d(branch3x3, int(384 * factor), [1, 3]),
                                                                      ops.conv2d(branch3x3, int(384 * factor), [3, 1])])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(448 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(384 * factor), [3, 3])
                                branch3x3dbl = tf.concat(axis=3,
                                                         values=[ops.conv2d(branch3x3dbl, int(384 * factor), [1, 3]),
                                                                 ops.conv2d(branch3x3dbl, int(384 * factor), [3, 1])])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
                            end_points['mixed_8x8x2048b'] = net
                            activations.append(net)
                        # Final pooling and prediction
                        with tf.variable_scope('logits'):
                            shape = net.get_shape()
                            net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
                            # 1 x 1 x 2048
                            net = ops.dropout(net, dropout_keep_prob, scope='dropout')
                            net = ops.flatten(net, scope='flatten')
                            # 2048
                            logits = ops.fc(net, num_classes, activation=None, scope='logits',
                                            restore=restore_logits)
                            # 1000
                            activations.append(logits)
                            end_points['logits'] = logits
                            end_points['predictions'] = tf.nn.softmax(logits, name='predictions')

                    return logits, activations  # , end_points


def inception_v3_test(inputs, opt, select, perturbation_params, perturbation_type, idx_gpu,
                      factor=1,
                      factor_end = 1,
                      dropout_keep_prob=0.8,
                      num_classes=1000,
                      is_training=False,
                      restore_logits=True,
                      scope=None):
    """Latest Inception from http://arxiv.org/abs/1512.00567.

      "Rethinking the Inception Architecture for Computer Vision"

      Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
      Zbigniew Wojna

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      dropout_keep_prob: dropout keep_prob.
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      restore_logits: whether or not the logits layers should be restored.
        Useful for fine-tuning a model with different num_classes.
      scope: Optional scope for name_scope.

    Returns:
      a list containing 'logits', 'aux_logits' Tensors.
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }

    end_points = {}

    with scopes.arg_scope([ops.conv2d, ops.fc], weight_decay=0.00004):
        with scopes.arg_scope([ops.conv2d],
                              stddev=0.1,
                              activation=tf.nn.relu,
                              batch_norm_params=batch_norm_params):
            with tf.name_scope(scope, 'inception_v3', [inputs]):
                with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                                      is_training=is_training):
                    with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                          stride=1, padding='VALID'):
                        # 299 x 299 x 3
                        end_points['conv0'] = ops.conv2d(inputs, int(32 * factor), [3, 3], stride=2,
                                                         scope='conv0')

                        activations_tmp = end_points['conv0']
                        if perturbation_type == 3:
                            batch_num = activations_tmp.get_shape().as_list()[0]
                            activations_tmp = pt.activation_noise(activations_tmp,
                                                                  perturbation_params[3][0], batch_num)
                        elif perturbation_type in [2, 4]:
                            ss = tf.reshape(
                                tf.tile(select[0], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                    opt.hyper.batch_size]),
                                [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                 int(activations_tmp.get_shape()[3])])
                            if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                begin = int(idx_gpu * batch_num)
                                size = activations_tmp.get_shape().as_list()
                                size[0] = batch_num
                                ss = tf.slice(ss, [begin, 0, 0, 0], size)
                            activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                          perturbation_params[4][0], ss)
                        end_points['conv0'] = activations_tmp

                        # 149 x 149 x 32
                        end_points['conv1'] = ops.conv2d(end_points['conv0'], int(32 * factor), [3, 3],
                                                         scope='conv1')

                        activations_tmp = end_points['conv1']
                        if perturbation_type == 3:
                            batch_num = activations_tmp.get_shape().as_list()[0]
                            activations_tmp = pt.activation_noise(activations_tmp,
                                                                  perturbation_params[3][1], batch_num)
                        elif perturbation_type in [2, 4]:
                            ss = tf.reshape(
                                tf.tile(select[1], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                    opt.hyper.batch_size]),
                                [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                 int(activations_tmp.get_shape()[3])])
                            if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                begin = int(idx_gpu * batch_num)
                                size = activations_tmp.get_shape().as_list()
                                size[0] = batch_num
                                ss = tf.slice(ss, [begin, 0, 0, 0], size)
                            activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                          perturbation_params[4][1], ss)
                        end_points['conv1'] = activations_tmp

                        # 147 x 147 x 32
                        end_points['conv2'] = ops.conv2d(end_points['conv1'], int(64 * factor), [3, 3],
                                                         padding='SAME', scope='conv2')

                        activations_tmp = end_points['conv2']
                        if perturbation_type == 3:
                            batch_num = activations_tmp.get_shape().as_list()[0]
                            activations_tmp = pt.activation_noise(activations_tmp,
                                                                  perturbation_params[3][2], batch_num)
                        elif perturbation_type in [2, 4]:
                            ss = tf.reshape(
                                tf.tile(select[2], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                    opt.hyper.batch_size]),
                                [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                 int(activations_tmp.get_shape()[3])])
                            if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                begin = int(idx_gpu * batch_num)
                                size = activations_tmp.get_shape().as_list()
                                size[0] = batch_num
                                ss = tf.slice(ss, [begin, 0, 0, 0], size)
                            activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                          perturbation_params[4][2], ss)
                        end_points['conv2'] = activations_tmp

                        # 147 x 147 x 64
                        end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                                           stride=2, scope='pool1')
                        # 73 x 73 x 64
                        end_points['conv3'] = ops.conv2d(end_points['pool1'], int(80 * factor), [1, 1],
                                                         scope='conv3')

                        activations_tmp = end_points['conv3']
                        if perturbation_type == 3:
                            batch_num = activations_tmp.get_shape().as_list()[0]
                            activations_tmp = pt.activation_noise(activations_tmp,
                                                                  perturbation_params[3][3], batch_num)
                        elif perturbation_type in [2, 4]:
                            ss = tf.reshape(
                                tf.tile(select[3], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                    opt.hyper.batch_size]),
                                [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                 int(activations_tmp.get_shape()[3])])
                            if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                begin = int(idx_gpu * batch_num)
                                size = activations_tmp.get_shape().as_list()
                                size[0] = batch_num
                                ss = tf.slice(ss, [begin, 0, 0, 0], size)
                            activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                          perturbation_params[4][3], ss)
                        end_points['conv3'] = activations_tmp

                        # 73 x 73 x 80.
                        end_points['conv4'] = ops.conv2d(end_points['conv3'], int(192 * factor), [3, 3],
                                                         scope='conv4')

                        activations_tmp = end_points['conv4']
                        if perturbation_type == 3:
                            batch_num = activations_tmp.get_shape().as_list()[0]
                            activations_tmp = pt.activation_noise(activations_tmp,
                                                                  perturbation_params[3][4], batch_num)
                        elif perturbation_type in [2, 4]:
                            ss = tf.reshape(
                                tf.tile(select[4], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                    opt.hyper.batch_size]),
                                [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                 int(activations_tmp.get_shape()[3])])
                            if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                begin = int(idx_gpu * batch_num)
                                size = activations_tmp.get_shape().as_list()
                                size[0] = batch_num
                                ss = tf.slice(ss, [begin, 0, 0, 0], size)
                            activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                          perturbation_params[4][4], ss)
                        end_points['conv4'] = activations_tmp

                        # 71 x 71 x 192.
                        end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                                           stride=2, scope='pool2')
                        # 35 x 35 x 192.
                        net = end_points['pool2']

                    # Inception blocks
                    with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                          stride=1, padding='SAME'):
                        # mixed: 35 x 35 x 256.
                        with tf.variable_scope('mixed_35x35x256a'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(64 * factor), [1, 1])
                            with tf.variable_scope('branch5x5'):
                                branch5x5 = ops.conv2d(net, int(48 * factor), [1, 1])
                                branch5x5 = ops.conv2d(branch5x5, int(64 * factor), [5, 5])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(64 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(32 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][5], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[5], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][5], ss)
                            end_points['mixed_35x35x256a'] = activations_tmp
                            net = activations_tmp

                        # mixed_1: 35 x 35 x 288.
                        with tf.variable_scope('mixed_35x35x288a'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(64 * factor), [1, 1])
                            with tf.variable_scope('branch5x5'):
                                branch5x5 = ops.conv2d(net, int(48 * factor), [1, 1])
                                branch5x5 = ops.conv2d(branch5x5, int(64 * factor), [5, 5])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(64 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(64 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][6], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[6], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][6], ss)
                            end_points['mixed_35x35x288a'] = activations_tmp
                            net = activations_tmp

                        # mixed_2: 35 x 35 x 288.
                        with tf.variable_scope('mixed_35x35x288b'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(64 * factor * factor_end), [1, 1])
                            with tf.variable_scope('branch5x5'):
                                branch5x5 = ops.conv2d(net, int(48 * factor * factor_end), [1, 1])
                                branch5x5 = ops.conv2d(branch5x5, int(64 * factor * factor_end), [5, 5])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(64 * factor * factor_end), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor * factor_end), [3, 3])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor * factor_end), [3, 3])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(64 * factor * factor_end), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][7], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[7], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][7], ss)
                            end_points['mixed_35x35x288b'] = activations_tmp
                            net = activations_tmp

                        # mixed_3: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768a'):
                            with tf.variable_scope('branch3x3'):
                                branch3x3 = ops.conv2d(net, int(384 * factor), [3, 3], stride=2, padding='VALID')
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(64 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(96 * factor), [3, 3],
                                                          stride=2, padding='VALID')
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                            net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][8], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[8], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][8], ss)
                            end_points['mixed_17x17x768a'] = activations_tmp
                            net = activations_tmp

                        # mixed4: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768b'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(192 * factor), [1, 1])
                            with tf.variable_scope('branch7x7'):
                                branch7x7 = ops.conv2d(net, int(128 * factor), [1, 1])
                                branch7x7 = ops.conv2d(branch7x7, int(128 * factor), [1, 7])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [7, 1])
                            with tf.variable_scope('branch7x7dbl'):
                                branch7x7dbl = ops.conv2d(net, int(128 * factor), [1, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(128 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(128 * factor), [1, 7])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(128 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][9], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[9], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][9], ss)
                            end_points['mixed_17x17x768b'] = activations_tmp
                            net = activations_tmp

                        # mixed_5: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768c'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(192 * factor), [1, 1])
                            with tf.variable_scope('branch7x7'):
                                branch7x7 = ops.conv2d(net, int(160 * factor), [1, 1])
                                branch7x7 = ops.conv2d(branch7x7, int(160 * factor), [1, 7])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [7, 1])
                            with tf.variable_scope('branch7x7dbl'):
                                branch7x7dbl = ops.conv2d(net, int(160 * factor), [1, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [1, 7])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][10], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[10], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][10], ss)
                            end_points['mixed_17x17x768c'] = activations_tmp
                            net = activations_tmp

                        # mixed_6: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768d'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(192 * factor), [1, 1])
                            with tf.variable_scope('branch7x7'):
                                branch7x7 = ops.conv2d(net, int(160 * factor), [1, 1])
                                branch7x7 = ops.conv2d(branch7x7, int(160 * factor), [1, 7])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [7, 1])
                            with tf.variable_scope('branch7x7dbl'):
                                branch7x7dbl = ops.conv2d(net, int(160 * factor), [1, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [1, 7])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(160 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][11], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[11], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][11], ss)
                            end_points['mixed_17x17x768d'] = activations_tmp
                            net = activations_tmp

                        # mixed_7: 17 x 17 x 768.
                        with tf.variable_scope('mixed_17x17x768e'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(192 * factor), [1, 1])
                            with tf.variable_scope('branch7x7'):
                                branch7x7 = ops.conv2d(net, int(192 * factor), [1, 1])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [1, 7])
                                branch7x7 = ops.conv2d(branch7x7, int(192 * factor), [7, 1])
                            with tf.variable_scope('branch7x7dbl'):
                                branch7x7dbl = ops.conv2d(net, int(192 * factor), [1, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [7, 1])
                                branch7x7dbl = ops.conv2d(branch7x7dbl, int(192 * factor), [1, 7])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][12], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[12], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][12], ss)
                            end_points['mixed_17x17x768e'] = activations_tmp
                            net = activations_tmp

                        # Auxiliary Head logits
                        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
                        with tf.variable_scope('aux_logits'):
                            aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
                                                      padding='VALID')
                            aux_logits = ops.conv2d(aux_logits, int(128 * factor), [1, 1], scope='proj')
                            # Shape of feature map before the final layer.
                            shape = aux_logits.get_shape()
                            aux_logits = ops.conv2d(aux_logits, int(768 * factor), shape[1:3], stddev=0.01,
                                                    padding='VALID')
                            aux_logits = ops.flatten(aux_logits)
                            aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                                                stddev=0.001, restore=restore_logits)
                            end_points['aux_logits'] = aux_logits
                        # mixed_8: 8 x 8 x 1280.
                        # Note that the scope below is not changed to not void previous
                        # checkpoints.
                        # (TODO) Fix the scope when appropriate.
                        with tf.variable_scope('mixed_17x17x1280a'):
                            with tf.variable_scope('branch3x3'):
                                branch3x3 = ops.conv2d(net, int(192 * factor), [1, 1])
                                branch3x3 = ops.conv2d(branch3x3, int(320 * factor), [3, 3], stride=2,
                                                       padding='VALID')
                            with tf.variable_scope('branch7x7x3'):
                                branch7x7x3 = ops.conv2d(net, int(192 * factor), [1, 1])
                                branch7x7x3 = ops.conv2d(branch7x7x3, int(192 * factor), [1, 7])
                                branch7x7x3 = ops.conv2d(branch7x7x3, int(192 * factor), [7, 1])
                                branch7x7x3 = ops.conv2d(branch7x7x3, int(192 * factor), [3, 3],
                                                         stride=2, padding='VALID')
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                            net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][13], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[13], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][13], ss)
                            end_points['mixed_17x17x1280a'] = activations_tmp
                            net = activations_tmp

                        # mixed_9: 8 x 8 x 2048.
                        with tf.variable_scope('mixed_8x8x2048a'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(320 * factor), [1, 1])
                            with tf.variable_scope('branch3x3'):
                                branch3x3 = ops.conv2d(net, int(384 * factor), [1, 1])
                                branch3x3 = tf.concat(axis=3, values=[ops.conv2d(branch3x3, int(384 * factor), [1, 3]),
                                                                      ops.conv2d(branch3x3, int(384 * factor), [3, 1])])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(448 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(384 * factor), [3, 3])
                                branch3x3dbl = tf.concat(axis=3,
                                                         values=[ops.conv2d(branch3x3dbl, int(384 * factor), [1, 3]),
                                                                 ops.conv2d(branch3x3dbl, int(384 * factor), [3, 1])])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][14], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[14], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][14], ss)
                            end_points['mixed_8x8x2048a'] = activations_tmp
                            net = activations_tmp

                        # mixed_10: 8 x 8 x 2048.
                        with tf.variable_scope('mixed_8x8x2048b'):
                            with tf.variable_scope('branch1x1'):
                                branch1x1 = ops.conv2d(net, int(320 * factor), [1, 1])
                            with tf.variable_scope('branch3x3'):
                                branch3x3 = ops.conv2d(net, int(384 * factor), [1, 1])
                                branch3x3 = tf.concat(axis=3, values=[ops.conv2d(branch3x3, int(384 * factor), [1, 3]),
                                                                      ops.conv2d(branch3x3, int(384 * factor), [3, 1])])
                            with tf.variable_scope('branch3x3dbl'):
                                branch3x3dbl = ops.conv2d(net, int(448 * factor), [1, 1])
                                branch3x3dbl = ops.conv2d(branch3x3dbl, int(384 * factor), [3, 3])
                                branch3x3dbl = tf.concat(axis=3,
                                                         values=[ops.conv2d(branch3x3dbl, int(384 * factor), [1, 3]),
                                                                 ops.conv2d(branch3x3dbl, int(384 * factor), [3, 1])])
                            with tf.variable_scope('branch_pool'):
                                branch_pool = ops.avg_pool(net, [3, 3])
                                branch_pool = ops.conv2d(branch_pool, int(192 * factor), [1, 1])
                            net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])

                            activations_tmp = net
                            if perturbation_type == 3:
                                batch_num = activations_tmp.get_shape().as_list()[0]
                                activations_tmp = pt.activation_noise(activations_tmp,
                                                                      perturbation_params[3][15], batch_num)
                            elif perturbation_type in [2, 4]:
                                ss = tf.reshape(
                                    tf.tile(select[15], [int(np.prod(activations_tmp.get_shape()[1:3])) *
                                                        opt.hyper.batch_size]),
                                    [-1, int(activations_tmp.get_shape()[1]), int(activations_tmp.get_shape()[2]),
                                     int(activations_tmp.get_shape()[3])])
                                if idx_gpu != -1:  # by default, it's -1 saying that its not using multi gpus
                                    batch_num = activations_tmp.get_shape().as_list()[0]
                                    begin = int(idx_gpu * batch_num)
                                    size = activations_tmp.get_shape().as_list()
                                    size[0] = batch_num
                                    ss = tf.slice(ss, [begin, 0, 0, 0], size)
                                activations_tmp = pt.activation_knockout_mask(activations_tmp,
                                                                              perturbation_params[4][15], ss)
                            end_points['mixed_8x8x2048b'] = activations_tmp
                            net = activations_tmp

                        # Final pooling and prediction
                        with tf.variable_scope('logits'):
                            shape = net.get_shape()
                            net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
                            # 1 x 1 x 2048
                            net = ops.dropout(net, dropout_keep_prob, scope='dropout')
                            net = ops.flatten(net, scope='flatten')
                            # 2048
                            logits = ops.fc(net, num_classes, activation=None, scope='logits',
                                            restore=restore_logits)
                            # 1000
                            end_points['logits'] = logits
                            end_points['predictions'] = tf.nn.softmax(logits, name='predictions')

                    return logits  # , end_points


def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
    """Yields the scope with the default parameters for inception_v3.

    Args:
      weight_decay: the weight decay for weights variables.
      stddev: standard deviation of the truncated guassian weight distribution.
      batch_norm_decay: decay for the moving average of batch_norm momentums.
      batch_norm_epsilon: small float added to variance to avoid dividing by zero.

    Yields:
      a arg_scope with the parameters needed for inception_v3.
    """
    # Set weight_decay for weights in Conv and FC layers.
    with scopes.arg_scope([ops.conv2d, ops.fc],
                          weight_decay=weight_decay):
        # Set stddev, activation and parameters for batch_norm.
        with scopes.arg_scope([ops.conv2d],
                              stddev=stddev,
                              activation=tf.nn.relu,
                              batch_norm_params={
                                  'decay': batch_norm_decay,
                                  'epsilon': batch_norm_epsilon}) as arg_scope:
            yield arg_scope
