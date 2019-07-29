import numpy as np
import tensorflow as tf


def reformat_weights(weights, input_width, input_height, filter_width, filter_height, in_channels, out_channels,
                     padding=0, stride=1):
    assert len(weights.shape) == 4
    # [filter_width, filter_height, in_channels, out_channels] = weights.shape
    # print(weights.shape)

    # compressed_weights = np.reshape(weights, [1, np.prod([filter_width, filter_height, in_channels]), out_channels])
    compressed_weights = tf.reshape(weights, [1, np.prod([filter_width, filter_height, in_channels]), out_channels])

    output_width = (input_width - filter_width + padding) / stride
    output_height = (input_height - filter_height + padding) / stride

    # repeated_weights = np.repeat(compressed_weights, axis=0, repeats=output_width * output_height)
    repeated_weights = tf.tile(compressed_weights, [np.int64(output_width * output_height), 1, 1])
    # print(repeated_weights.shape)

    return repeated_weights
