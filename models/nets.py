import tensorflow as tf

from models.mlp import MLP1 as net_MLP1
from models.mlp import MLP1_test as net_MLP1_test
from models.mlp import MLP1_linear as net_MLP1_linear
from models.mlp import MLP1_linear_test as net_MLP1_linear_test
from models.mlp import MLP3 as net_MLP3
from models.mlp import MLP3_test as net_MLP3_test
from models.alexnet import Alexnet as net_Alexnet
from models.alexnet import Alexnet_test as net_Alexnet_test
from models.resnet_cifar import ResNet as net_ResNet_cifar
from models.resnet_cifar import ResNet_test as net_ResNet_cifar_test
from models.resnet_imagenet import ResNet as net_ResNet_imagenet
from utils import summary as summ


def MLP3(x, dropout_rate, opt, labels_id):
    return net_MLP3(x, opt)

def MLP3_test(x, dropout_rate, select, opt, labels_id, perturbation_params, perturbation_type):
    return net_MLP3_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type)

def MLP1(x, dropout_rate, opt, labels_id):
    return net_MLP1(x, opt)


def MLP1_test(x, dropout_rate, select, opt, labels_id, perturbation_params, perturbation_type):
    return net_MLP1_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type)

def MLP1_linear(x, dropout_rate, opt, labels_id):
    return net_MLP1_linear(x, opt)

def MLP1_linear_test(x, dropout_rate, select, opt, labels_id, perturbation_params, perturbation_type):
    return net_MLP1_linear_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type)


def Alexnet(x, dropout_rate, opt, labels_id):
    return net_Alexnet(x, opt, labels_id, dropout_rate)


def Alexnet_test(x, dropout_rate, select, opt, labels_id, perturbation_params, perturbation_type):
    return net_Alexnet_test(x, opt, select, labels_id, dropout_rate, perturbation_params, perturbation_type)


# ignore dropout rate and labels_id in the ResNet models
def ResNet_cifar(x, dropout_rate, opt, labels_id):
    return net_ResNet_cifar(x, opt)


def ResNet_cifar_test(x, dropout_rate, select, opt, labels_id, perturbation_params, perturbation_type):
    return net_ResNet_cifar_test(x, opt, select, perturbation_params, perturbation_type)
