import tensorflow as tf

def resnet(im, opt):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

    return im - _CHANNEL_MEANS


def inception(im, opt):
    image = tf.divide(im, 255)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image