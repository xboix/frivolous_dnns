import numpy as np
import sys
import datasets
import copy
import pickle


class DNN(object):

    def __init__(self):
        self.name = "ResNet"
        self.layers = 5  # not including the logits output layer
        self.factor = 1
        self.factor_end = 1



class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 3072


class Experiments(object):

    def __init__(self, id, dataset, output_path, family_id, family_name,
                 results_dir='/om/user/scasper/workspace/models/resnet_imagenet/',
                 csv_dir='/om/user/scasper/workspace/csvs/resnet_imagenet/'):
        self.log_dir_base = output_path
        self.results_dir = results_dir
        self.csv_dir = csv_dir

        # Recordings
        self.max_to_keep_checkpoints = 2

        # Test after training:
        self.skip_train = False

        # Start from scratch even if it existed
        self.restart = False

        # Skip running experiments
        self.skip = False

        # Save extense summary
        self.extense_summary = True

        self.ID = id
        self.name = "ID" + str(id)

        self.family_ID = family_id
        self.family_name = family_name

        # Add additional descriptors to Experiments
        self.dataset = dataset
        self.dnn = DNN()
        self.hyper = Hyperparameters()

        self.time_step = 0


def get_experiments(output_path, dataset_path):

    opt_data = datasets.get_datasets(dataset_path)

    # # #
    # Create set of experiments
    opt = []

    idx_base = 0
    idx_family = 0

    # RESNETS

    # Maximum batch size per size
    # id 0-4
    for n_multiplier, batch_size in zip([0.25, 0.5, 1, 2, 4], [8192, 4096, 3072, 1024, 512]):
        opt_handle = Experiments(id=idx_base,
                                 dataset=opt_data[0], output_path=output_path,
                                 family_id=idx_family, family_name="ResNet18")

        opt_handle.skip_train = False
        opt_handle.dnn.name = "resnet"
        opt_handle.dnn.factor = n_multiplier
        opt_handle.hyper.batch_size = batch_size
        opt += [copy.deepcopy(opt_handle)]
        idx_base += 1

    # batch size = 4096
    ### 0.5 already calculated!
    # id 5
    for n_multiplier in [0.25]:
        opt_handle = Experiments(id=idx_base,
                                 dataset=opt_data[0], output_path=output_path,
                                 family_id=idx_family, family_name="ResNet18")

        opt_handle.skip_train = False
        opt_handle.dnn.name = "resnet"
        opt_handle.dnn.factor = n_multiplier
        opt_handle.hyper.batch_size = 4096
        opt += [copy.deepcopy(opt_handle)]
        idx_base += 1

    # batch size = 2048
    # id 6 - 8
    for n_multiplier in [0.25, 0.5, 1]:
        opt_handle = Experiments(id=idx_base,
                                 dataset=opt_data[0], output_path=output_path,
                                 family_id=idx_family, family_name="ResNet18")

        opt_handle.skip_train = False
        opt_handle.dnn.name = "resnet"
        opt_handle.dnn.factor = n_multiplier
        opt_handle.hyper.batch_size = 2048
        opt += [copy.deepcopy(opt_handle)]
        idx_base += 1

    # batch size = 1024
    ### 2 already calculated!
    # id 9 - 11
    for n_multiplier in [0.25, 0.5, 1]:
        opt_handle = Experiments(id=idx_base,
                                 dataset=opt_data[0], output_path=output_path,
                                 family_id=idx_family, family_name="ResNet18")

        opt_handle.skip_train = False
        opt_handle.dnn.name = "resnet"
        opt_handle.dnn.factor = n_multiplier
        opt_handle.hyper.batch_size = 1024
        opt += [copy.deepcopy(opt_handle)]
        idx_base += 1

    # INCEPTIONS

    # Maximum batch size per size
    # id 12-14
    for n_multiplier, batch_size in zip([1, 0.25, 0.5], [512, 512, 512]):
        opt_handle = Experiments(id=idx_base,
                                 dataset=opt_data[0], output_path=output_path,
                                 family_id=idx_family, family_name="Inception_v3")

        opt_handle.skip_train = False
        opt_handle.dnn.name = "inception"
        opt_handle.dnn.factor = n_multiplier
        opt_handle.hyper.batch_size = batch_size
        opt_handle.dnn.layers = 16  # not including the logits output layer # 16 might not be right...
        opt_handle.results_dir = '/om/user/scasper/workspace/models/inception_imagenet/'
        opt_handle.csv_dir = '/om/user/scasper/workspace/csvs/inception_imagenet/'
        opt += [copy.deepcopy(opt_handle)]
        idx_base += 1

    # Maximum batch size per size
    # id 15-18
    for n_multiplier, batch_size in zip([0.25, 0.5, 2, 4], [512, 512, 512, 512]):
        opt_handle = Experiments(id=idx_base,
                                 dataset=opt_data[0], output_path=output_path,
                                 family_id=idx_family, family_name="Inception_v3")

        opt_handle.skip_train = False
        opt_handle.dnn.name = "inception"
        opt_handle.dnn.factor = 1
        opt_handle.dnn.factor_end = n_multiplier
        opt_handle.hyper.batch_size = batch_size
        opt_handle.dnn.layers = 16  # not including the logits output layer # 16 might not be right...
        opt_handle.results_dir = '/om/user/scasper/workspace/models/inception_imagenet/'
        opt_handle.csv_dir = '/om/user/scasper/workspace/csvs/inception_imagenet/'
        opt += [copy.deepcopy(opt_handle)]
        idx_base += 1

    # RESNET timesteps
    # id 19-38
    for step in range(4):
        for n_multiplier, batch_size in zip([0.25, 0.5, 1, 2, 4], [8192, 4096, 3072, 1024, 512]):
            opt_handle = Experiments(id=idx_base,
                                     dataset=opt_data[0], output_path=output_path,
                                     family_id=idx_family, family_name="ResNet18")

            opt_handle.time_step = step
            opt_handle.skip_train = False
            opt_handle.dnn.name = "resnet"
            opt_handle.dnn.factor = n_multiplier
            opt_handle.hyper.batch_size = batch_size
            opt += [copy.deepcopy(opt_handle)]
            idx_base += 1



    print('OPTS LOOKUP:')
    for ID in range(len(opt)):
        print('ID: ' + str(ID) + ', ' + str(opt[ID].dnn.name) + ', factor: ' +
              str(opt[ID].dnn.factor) + ', batch_size:' + str(opt[ID].hyper.batch_size))

    # print("Number of experiments:", len(opt))

    return opt
