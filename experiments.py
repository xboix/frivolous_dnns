import numpy as np

master_seed = 0


class Dataset(object):

    def __init__(self):

        # Dataset general
        self.dataset_path = '/om/user/scasper/data/cifar10/'
        self.proportion_training_set = .95
        self.shuffle_data = True

        # For reusing tfrecords:
        self.reuse_TFrecords = False
        self.reuse_TFrecords_ID = 0
        self.reuse_TFrecords_path = ""

        # Set random labels
        self.random_labels = False
        self.scramble_data = False

        # Transfer learning
        self.transfer_learning = False
        self.transfer_pretrain = True
        self.transfer_label_offset = 0
        self.transfer_restart_name = "_pretrain"
        self.transfer_append_name = ""

    # Set base tfrecords
    def generate_base_tfrecords(self):
        self.reuse_TFrecords = False

    # Set reuse tfrecords mode
    def reuse_tfrecords(self, experiment):
        self.reuse_TFrecords = True
        self.reuse_TFrecords_ID = experiment.ID
        self.reuse_TFrecords_path = experiment.name

    # Transfer learning
    def do_pretrain_transfer_learning(self):
        self.transfer_learning = True
        self.transfer_append_name = self.transfer_restart_name

        # def do_transfer_transfer_learning(self):
        self.transfer_learning = True
        self.transfer_pretrain = True
        self.transfer_label_offset = 5
        self.transfer_append_name = "_transfer"


class DNN(object):

    def __init__(self):
        self.name = ''
        self.pretrained = False
        self.layers = 5
        # self.version = 1
        self.neuron_multiplier = np.ones(self.layers-1)

    # def set_num_layers(self, num_layers):
    #     self.layers = num_layers
    #     self.neuron_multiplier = np.ones([self.layers])


class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.lr_bs_factor = 1  # factor by which lr and batch_size are multiplied by
        self.num_epochs_per_decay = 1
        self.learning_rate_factor_per_decay = 0.95
        self.weight_decay = 0
        self.max_num_epochs = 500
        self.crop_size = 28
        self.image_size = 32
        self.drop_train = 1
        self.drop_test = 1
        self.momentum = 0.9
        self.augmentation = False
        self.init_factor = False


class Experiments(object):

    def __init__(self, id, name):
        self.log_dir_base = ''
        self.csv_dir = ''
        self.seed = master_seed

        # Training
        # 0=glorot norm, 1=glorot unif, 2=he norm, 3=he unif, 4=lecun norm, 5=lecun unif; invalid defaults 0
        self.init_type = 0
        # 0=momentum, 1=sgd, 2=adam, 3=adagrad, 4=proximal adagrad, 5=rmsprop, 6=FTRL
        self.optimizer = 0
        # 0=relu, 1=leaky relu, 2=elu, 3=exponential, 4=sigmoid, 5=tanh
        self.act_function = 0

        # Recordings
        self.max_to_keep_checkpoints = 5
        self.recordings = False
        self.num_batches_recordings = 0

        # Plotting
        self.plot_details = 0
        self.plotting = False

        # Test after training:
        self.test = False

        # Start from scratch even if it existed
        self.restart = False

        # Skip running experiments
        self.skip = False

        # Save extense summary
        self.extense_summary = True

        # Add ID to name:
        self.ID = id
        self.name = 'ID' + str(self.ID) + "_" + name

        # Add additional descriptors to Experiments
        self.dataset = Dataset()
        self.dnn = DNN()
        self.hyper = Hyperparameters()

    def do_recordings(self, max_epochs):
        self.max_to_keep_checkpoints = 0
        self.recordings = True
        self.hyper.max_num_epochs = max_epochs
        self.num_batches_recordings = 10

    def do_plotting(self, plot_details=0):
        self.plot_details = plot_details
        self.plotting = True


# Create set of experiments
opt = []

name = ['Alexnet']
initialization_type = [0, 1, 2, 3, 4, 5]
initialization_multiplier = [1.0]  # [0.1, 0.5, 1, 5, 10]
neuron_multiplier = [0.25, 0.5, 1, 2, 4]
training_prop = [1]  # , 0.5, 0.25, 0.125, 0.0625]
learning_rate = [1e-2]  # [1e-1, 1e-2, 1e-3]
regularizers = [0]  # 1, 2, 3, 4]
flag_random = [0, 1]

##############################################################################################

# experiments 1 and 2 are to make the base tfrecords files that the other modesl use

idx = 0
for p in training_prop:
    # Create base for TF records:
    opt += [Experiments(idx, "data_" + str(p))]
    opt[-1].dataset.proportion_training_set *= p
    opt[-1].hyper.max_num_epochs = 0
    idx += 1

idx_base_random = idx
for p in training_prop:
    # Create base for TF records:
    opt += [Experiments(idx, "data_random")]
    opt[-1].hyper.max_num_epochs = 0
    opt[-1].dataset.proportion_training_set *= p
    opt[-1].dataset.random_labels = True
    idx += 1

##############################################################################################

# experiments 2-61 are the init_scheme_tests

for nn_name in name:
    for flag_rand in flag_random:
        for init_type in initialization_type:
            for init_mult in initialization_multiplier:
                for neuron_mult in neuron_multiplier:
                    for train_prop_idx in range(len(training_prop)):
                        for lr in learning_rate:
                            for reg in regularizers:

                                if flag_rand and reg > 0:
                                    continue

                                opt.append(Experiments(idx, nn_name + '_itype=' + str(init_type) + '_nmult=' +
                                                       str(neuron_mult) + '_reg=' + str(reg) + '_rand=' +
                                                       str(flag_rand)))

                                opt[-1].log_dir_base = '/om/user/scasper/workspace/models/init_scheme/'
                                opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/init_scheme/'
                                opt[-1].dnn.name = nn_name
                                opt[-1].init_type = init_type
                                opt[-1].hyper.init_factor = init_mult
                                opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                opt[-1].hyper.learning_rate = lr
                                opt[-1].hyper.num_epochs_per_decay = \
                                    int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])

                                if flag_rand == 0:
                                    opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                    opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                else:
                                    opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                    opt[-1].dataset.random_labels = True
                                    opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                                                       training_prop[train_prop_idx])

                                if reg == 1:
                                    opt[-1].hyper.augmentation = True
                                    opt[-1].hyper.max_num_epochs *= int(2)
                                elif reg == 2:
                                    opt[-1].hyper.drop_train = 0.5
                                elif reg == 3:
                                    opt[-1].hyper.weight_decay = 0.001
                                elif reg == 4:
                                    opt[-1].hyper.augmentation = True
                                    opt[-1].hyper.max_num_epochs *= int(2)
                                    opt[-1].hyper.drop_train = 0.5
                                    opt[-1].hyper.weight_decay = 0.001
                                idx += 1

##############################################################################################

# experiments 62-91 are the main tests

initialization_type = [0]
regularizers = [0, 1, 2, 3, 4]

for nn_name in name:
    for init_type in initialization_type:
        for init_mult in initialization_multiplier:
            for neuron_mult in neuron_multiplier:
                for train_prop_idx in range(len(training_prop)):
                    for lr in learning_rate:
                        for reg in regularizers:
                            for flag_rand in flag_random:

                                if flag_rand and reg > 0:
                                    continue

                                opt.append(Experiments(idx, nn_name + '_itype=' + str(init_type) + '_nmult=' +
                                                       str(neuron_mult) + '_reg=' + str(reg) + '_rand=' +
                                                       str(flag_rand) + '_seed=' + str(master_seed)))

                                opt[-1].log_dir_base = '/om/user/scasper/workspace/models/replication/'
                                opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/replication/'
                                opt[-1].dnn.name = nn_name
                                opt[-1].init_type = init_type
                                opt[-1].hyper.init_factor = init_mult
                                opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                opt[-1].hyper.learning_rate = lr
                                opt[-1].hyper.num_epochs_per_decay = \
                                    int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])

                                if flag_rand == 0:
                                    opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                    opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                else:
                                    opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                    opt[-1].dataset.random_labels = True
                                    opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                                                       training_prop[train_prop_idx])

                                if reg == 1:
                                    opt[-1].hyper.augmentation = True
                                    opt[-1].hyper.max_num_epochs *= int(2)
                                elif reg == 2:
                                    opt[-1].hyper.drop_train = 0.5
                                elif reg == 3:
                                    opt[-1].hyper.weight_decay = 0.001
                                elif reg == 4:
                                    opt[-1].hyper.augmentation = True
                                    opt[-1].hyper.max_num_epochs *= int(2)
                                    opt[-1].hyper.drop_train = 0.5
                                    opt[-1].hyper.weight_decay = 0.001
                                idx += 1

##############################################################################################

# experiments 92-126 are the optimizer tests

regularizers = [0]
optimizers = [0, 1, 2, 3, 4, 5, 6]
flag_random = [0]

for nn_name in name:
    for init_type in initialization_type:
        for init_mult in initialization_multiplier:
            for neuron_mult in neuron_multiplier:
                for train_prop_idx in range(len(training_prop)):
                    for lr in learning_rate:
                        for reg in regularizers:
                            for flag_rand in flag_random:
                                for optim in optimizers:

                                    if flag_rand and reg > 0:
                                        continue

                                    opt.append(Experiments(idx, nn_name + '_nmult=' +
                                                           str(neuron_mult) + '_reg=' + str(reg) + '_rand=' +
                                                           str(flag_rand) + '_optim=' + str(optim) +
                                                           '_seed=' + str(master_seed)))

                                    opt[-1].log_dir_base = '/om/user/scasper/workspace/models/optimizers/'
                                    opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/optimizers/'
                                    opt[-1].dnn.name = nn_name
                                    opt[-1].init_type = init_type
                                    opt[-1].hyper.init_factor = init_mult
                                    opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                    opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                    opt[-1].hyper.learning_rate = lr
                                    opt[-1].hyper.num_epochs_per_decay = \
                                        int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])
                                    opt[-1].optimizer = optim

                                    if flag_rand == 0:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                        opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                    else:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                        opt[-1].dataset.random_labels = True
                                        opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                                                           training_prop[train_prop_idx])

                                    idx += 1

##############################################################################################

# experiments 127-156 are the activation function tests

act_functions = [0, 1, 2, 3, 4, 5]

for nn_name in name:
    for init_type in initialization_type:
        for init_mult in initialization_multiplier:
            for neuron_mult in neuron_multiplier:
                for train_prop_idx in range(len(training_prop)):
                    for lr in learning_rate:
                        for reg in regularizers:
                            for flag_rand in flag_random:
                                for af in act_functions:

                                    if flag_rand and reg > 0:
                                        continue

                                    opt.append(Experiments(idx, nn_name + '_nmult=' +
                                                           str(neuron_mult) + '_reg=' + str(reg) + '_rand=' +
                                                           str(flag_rand) + '_af=' + str(af) +
                                                           '_seed=' + str(master_seed)))

                                    opt[-1].log_dir_base = '/om/user/scasper/workspace/models/act_functions/'
                                    opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/act_functions/'
                                    opt[-1].dnn.name = nn_name
                                    opt[-1].init_type = init_type
                                    opt[-1].hyper.init_factor = init_mult
                                    opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                    opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                    opt[-1].hyper.learning_rate = lr
                                    opt[-1].hyper.num_epochs_per_decay = \
                                        int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])
                                    opt[-1].act_function = af

                                    if flag_rand == 0:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                        opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                    else:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                        opt[-1].dataset.random_labels = True
                                        opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                                                           training_prop[train_prop_idx])

                                    idx += 1

##############################################################################################

# experiments 157-216 are the tests for different factors to multiple the lr and batch size by

lr_bs_factors = [0.25, 4]
regularizers = [0, 1, 2, 3, 4]
flag_random = [0, 1]

for nn_name in name:
    for flag_rand in flag_random:
        for init_type in initialization_type:
            for init_mult in initialization_multiplier:
                for train_prop_idx in range(len(training_prop)):
                    for lr in learning_rate:
                        for reg in regularizers:
                            for neuron_mult in neuron_multiplier:
                                for lr_bs in lr_bs_factors:

                                    if flag_rand and reg > 0:
                                        continue

                                    opt.append(Experiments(idx, nn_name + '_nmult=' +
                                                           str(neuron_mult) + '_reg=' + str(reg) + '_lrbs=' +
                                                           str(lr_bs) + '_rand=' + str(flag_rand) + '_seed=' +
                                                           str(master_seed)))

                                    opt[-1].log_dir_base = '/om/user/scasper/workspace/models/lr_bs/'
                                    opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/lr_bs/'
                                    opt[-1].dnn.name = nn_name
                                    opt[-1].init_type = init_type
                                    opt[-1].hyper.init_factor = init_mult
                                    opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                    opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                    opt[-1].hyper.learning_rate = lr
                                    opt[-1].hyper.learning_rate *= lr_bs
                                    opt[-1].hyper.batch_size = int(opt[-1].hyper.batch_size * lr_bs)
                                    opt[-1].hyper.num_epochs_per_decay = \
                                        int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])

                                    if flag_rand == 0:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                        opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                    else:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                        opt[-1].dataset.random_labels = True
                                        opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                        training_prop[train_prop_idx])

                                    if reg == 1:
                                        opt[-1].hyper.augmentation = True
                                        opt[-1].hyper.max_num_epochs *= int(2)
                                    elif reg == 2:
                                        opt[-1].hyper.drop_train = 0.5
                                    elif reg == 3:
                                        opt[-1].hyper.weight_decay = 0.001
                                    elif reg == 4:
                                        opt[-1].hyper.augmentation = True
                                        opt[-1].hyper.max_num_epochs *= int(2)
                                        opt[-1].hyper.drop_train = 0.5
                                        opt[-1].hyper.weight_decay = 0.001
                                    idx += 1

##############################################################################################

# experiments 217-231 are the tests for resnet in cifar

name = ['ResNet_cifar']
regularizers = [0]
lr_bs_factors = [1, 0.25]
learning_rate = [1e-1]

for nn_name in name:
    for init_type in initialization_type:
        for init_mult in initialization_multiplier:
            for neuron_mult in neuron_multiplier:
                for train_prop_idx in range(len(training_prop)):
                    for lr in learning_rate:
                        for reg in regularizers:
                            for lr_bs in lr_bs_factors:
                                for flag_rand in flag_random:

                                    if flag_rand and lr_bs == 0.25:
                                        continue

                                    opt.append(Experiments(idx, nn_name + '_nmult=' +
                                                           str(neuron_mult) + '_lrbs=' +
                                                           str(lr_bs) + '_rand=' + str(flag_rand) + '_seed=' +
                                                           str(master_seed)))

                                    opt[-1].log_dir_base = '/om/user/scasper/workspace/models/resnet_cifar/'
                                    opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/resnet_cifar/'
                                    opt[-1].dnn.name = nn_name
                                    opt[-1].init_type = init_type
                                    opt[-1].hyper.init_factor = init_mult
                                    opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                    opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                    opt[-1].hyper.learning_rate = lr
                                    opt[-1].hyper.lr_bs_factor = lr_bs
                                    opt[-1].hyper.learning_rate *= lr_bs
                                    opt[-1].hyper.batch_size = int(opt[-1].hyper.batch_size * lr_bs)
                                    opt[-1].hyper.weight_decay = 2e-4
                                    opt[-1].hyper.num_epochs_per_decay = \
                                        int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])

                                    if flag_rand == 0:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                        opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                        opt[-1].hyper.augmentation = True
                                        opt[-1].hyper.weight_decay = 2e-4
                                    else:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                        opt[-1].dataset.random_labels = True
                                        opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                                                           training_prop[train_prop_idx])
                                        opt[-1].hyper.augmentation = False
                                        opt[-1].hyper.weight_decay = 0
                                    idx += 1

##############################################################################################

# experiments 232 and 233 need to be copies of 86 and 91 to help with the mappability experiment

regularizers = [0, 4]
flag_random = [0]
name = ['Alexnet']
neuron_multiplier = [4]
learning_rate = [1e-2]

for nn_name in name:
    for init_type in initialization_type:
        for init_mult in initialization_multiplier:
            for neuron_mult in neuron_multiplier:
                for train_prop_idx in range(len(training_prop)):
                    for lr in learning_rate:
                        for reg in regularizers:
                            for flag_rand in flag_random:

                                if flag_rand and reg > 0:
                                    continue

                                opt.append(Experiments(idx, nn_name + '_itype=' + str(init_type) + '_nmult=' +
                                                       str(neuron_mult) + '_reg=' + str(reg) + '_rand=' +
                                                       str(flag_rand) + '_seed=' + str(master_seed+1)))

                                opt[-1].log_dir_base = '/om/user/scasper/workspace/models/replication/'
                                opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/replication/'
                                opt[-1].dnn.name = nn_name
                                opt[-1].init_type = init_type
                                opt[-1].hyper.init_factor = init_mult
                                opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                opt[-1].hyper.learning_rate = lr
                                opt[-1].hyper.num_epochs_per_decay = \
                                    int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])
                                opt[-1].seed += 1

                                if flag_rand == 0:
                                    opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                    opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                else:
                                    opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                    opt[-1].dataset.random_labels = True
                                    opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                                                       training_prop[train_prop_idx])

                                if reg == 1:
                                    opt[-1].hyper.augmentation = True
                                    opt[-1].hyper.max_num_epochs *= int(2)
                                elif reg == 2:
                                    opt[-1].hyper.drop_train = 0.5
                                elif reg == 3:
                                    opt[-1].hyper.weight_decay = 0.001
                                elif reg == 4:
                                    opt[-1].hyper.augmentation = True
                                    opt[-1].hyper.max_num_epochs *= int(2)
                                    opt[-1].hyper.drop_train = 0.5
                                    opt[-1].hyper.weight_decay = 0.001
                                idx += 1

##############################################################################################

# experiments 234-238 are more tests for resnet in cifar which have a lr_bs_factor of 4

name = ['ResNet_cifar']
regularizers = [0]
lr_bs_factors = [4]
learning_rate = [1e-1]
neuron_multiplier = [0.25, 0.5, 1, 2, 4]

for nn_name in name:
    for init_type in initialization_type:
        for init_mult in initialization_multiplier:
            for neuron_mult in neuron_multiplier:
                for train_prop_idx in range(len(training_prop)):
                    for lr in learning_rate:
                        for reg in regularizers:
                            for lr_bs in lr_bs_factors:
                                for flag_rand in flag_random:

                                    if flag_rand and lr_bs == 0.25:
                                        continue

                                    opt.append(Experiments(idx, nn_name + '_nmult=' +
                                                           str(neuron_mult) + '_lrbs=' +
                                                           str(lr_bs) + '_rand=' + str(flag_rand) + '_seed=' +
                                                           str(master_seed)))

                                    opt[-1].log_dir_base = '/om/user/scasper/workspace/models/resnet_cifar/'
                                    opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/resnet_cifar/'
                                    opt[-1].dnn.name = nn_name
                                    opt[-1].init_type = init_type
                                    opt[-1].hyper.init_factor = init_mult
                                    opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                    opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                    opt[-1].hyper.learning_rate = lr
                                    opt[-1].hyper.lr_bs_factor = lr_bs
                                    opt[-1].hyper.learning_rate *= lr_bs
                                    opt[-1].hyper.batch_size = int(opt[-1].hyper.batch_size * lr_bs)
                                    opt[-1].hyper.weight_decay = 2e-4
                                    opt[-1].hyper.num_epochs_per_decay = \
                                        int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])

                                    if flag_rand == 0:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                        opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                        opt[-1].hyper.augmentation = True
                                    else:
                                        opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                        opt[-1].dataset.random_labels = True
                                        opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                                                           training_prop[train_prop_idx])
                                        opt[-1].hyper.augmentation = False
                                        opt[-1].hyper.weight_decay = 0
                                    idx += 1

##############################################################################################

# experiments 239 and 240 need to be copies of 62 and 67 to help with the mappability experiment

regularizers = [0, 4]
flag_random = [0]
name = ['Alexnet']
neuron_multiplier = [0.25]
learning_rate = [1e-2]

for nn_name in name:
    for init_type in initialization_type:
        for init_mult in initialization_multiplier:
            for neuron_mult in neuron_multiplier:
                for train_prop_idx in range(len(training_prop)):
                    for lr in learning_rate:
                        for reg in regularizers:
                            for flag_rand in flag_random:

                                if flag_rand and reg > 0:
                                    continue

                                opt.append(Experiments(idx, nn_name + '_itype=' + str(init_type) + '_nmult=' +
                                                       str(neuron_mult) + '_reg=' + str(reg) + '_rand=' +
                                                       str(flag_rand) + '_seed=' + str(master_seed+1)))

                                opt[-1].log_dir_base = '/om/user/scasper/workspace/models/replication/'
                                opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/replication/'
                                opt[-1].dnn.name = nn_name
                                opt[-1].init_type = init_type
                                opt[-1].hyper.init_factor = init_mult
                                opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
                                opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
                                opt[-1].hyper.learning_rate = lr
                                opt[-1].hyper.num_epochs_per_decay = \
                                    int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])
                                opt[-1].seed += 1

                                if flag_rand == 0:
                                    opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
                                    opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
                                else:
                                    opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx + idx_base_random])
                                    opt[-1].dataset.random_labels = True
                                    opt[-1].hyper.max_num_epochs = int(opt[-1].hyper.max_num_epochs * 10 /
                                                                       training_prop[train_prop_idx])

                                if reg == 1:
                                    opt[-1].hyper.augmentation = True
                                    opt[-1].hyper.max_num_epochs *= int(2)
                                elif reg == 2:
                                    opt[-1].hyper.drop_train = 0.5
                                elif reg == 3:
                                    opt[-1].hyper.weight_decay = 0.001
                                elif reg == 4:
                                    opt[-1].hyper.augmentation = True
                                    opt[-1].hyper.max_num_epochs *= int(2)
                                    opt[-1].hyper.drop_train = 0.5
                                    opt[-1].hyper.weight_decay = 0.001
                                idx += 1

##############################################################################################

# experiments X-X are ResNet18s in imagenet

# name = ['ResNet_imagenet']
# regularizers = [0]
# lr_bs_factors = [1]
# neuron_multiplier = [0.25, 0.5, 1, 2, 4]
# batch_sizes = [512, 1024, 2048, 4096, 8192]
# flag_random = [0]
#
# for nn_name in name:
#     for init_type in initialization_type:
#         for init_mult in initialization_multiplier:
#             for nm_idx in range(len(neuron_multiplier)):
#                 for bs_idx in range(len(batch_sizes)):
#                     for train_prop_idx in range(len(training_prop)):
#                         for reg in regularizers:
#                             for flag_rand in flag_random:
#
#                                 if nm_idx + bs_idx > 4:  # these ones were impossible to train
#                                     continue
#
#                                 neuron_mult = neuron_multiplier[nm_idx]
#                                 batch_size = batch_sizes[bs_idx]
#
#                                 opt.append(Experiments(idx, nn_name + '_nmult=' + str(neuron_mult) +
#                                                        '_bsize=' + str(batch_size) + '_seed=' + str(master_seed)))
#
#                                 opt[-1].log_dir_base = '/om/user/scasper/workspace/models/resnet_imagenet/'
#                                 opt[-1].csv_dir = '/om/user/scasper/workspace/csvs/resnet_imagenet/'
#                                 opt[-1].dnn.name = nn_name
#                                 opt[-1].init_type = init_type
#                                 opt[-1].hyper.init_factor = init_mult
#                                 opt[-1].dnn.layers = 19
#                                 opt[-1].dnn.neuron_multiplier = np.ones(opt[-1].dnn.layers)
#                                 opt[-1].dnn.neuron_multiplier.fill(neuron_mult)
#                                 opt[-1].hyper.batch_size = batch_size
#                                 opt[-1].dataset.proportion_training_set *= training_prop[train_prop_idx]
#                                 opt[-1].hyper.num_epochs_per_decay = \
#                                     int(opt[-1].hyper.num_epochs_per_decay / training_prop[train_prop_idx])
#
#                                 opt[-1].dataset.reuse_tfrecords(opt[train_prop_idx])
#                                 opt[-1].dataset.reuse_TFrecords_path = \
#                                     '/om/user/scasper/workspace/models/resnet_imagenet/data'
#                                 opt[-1].hyper.max_num_epochs //= training_prop[train_prop_idx]
#                                 opt[-1].hyper.augmentation = True
#                                 opt[-1].hyper.weight_decay = 1e-4
#
#                                 idx += 1

##############################################################################################


def write_lookup_file():
    with open('experiment_lookup.txt', 'w') as f:
        for i in range(len(opt)):
            f.write(str(i) + '\n')
            f.write(opt[i].name + '\n')
            f.write('\n')

# write_lookup_file()
