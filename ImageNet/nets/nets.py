from nets.resnet import ResNet as net_resenet
from nets.resnet import ResNet_test as net_resnet_test
from nets.inception.inception_model import inception_v3 as net_inception
from nets.inception.inception_model import inception_v3_test as net_inception_test


def resnet(x, opt):
    return net_resenet(x, opt)


def resnet_test(x, opt, select, perturbation_params, perturbation_type, idx_gpu=-1):
    return net_resnet_test(x, opt, select, perturbation_params, perturbation_type, idx_gpu)


def inception(x, opt):
    return net_inception(x, opt, opt.dnn.factor, opt.dnn.factor_end, num_classes=1001)


def inception_test(x, opt, select, perturbation_params, perturbation_type, idx_gpu=-1):
    return net_inception_test(x, opt, select, perturbation_params, perturbation_type, idx_gpu,
                         opt.dnn.factor, opt.dnn.factor_end, num_classes=1001)
