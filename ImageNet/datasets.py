import numpy as np
import sys
import copy

class Dataset(object):

    def __init__(self, name, dataset_path):

        # Dataset general
        self.dataset_path = ""
        self.num_images_training = 1281167
        self.num_images_testing = 0
        self.num_images_validation = 50000
        self.shuffle_data = True

        self.dataset_name = "ImageNet"

        self.log_dir_base = dataset_path

        self.log_name = name


def get_datasets(dataset_path):

    # Create set of experiments
    opt = []
    idx = 0

    for k, num_data in enumerate(['']):

        opt_handle = Dataset("ImageNet", dataset_path)
        opt_handle.num_images_training = num_data

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    return opt

