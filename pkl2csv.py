
from __future__ import print_function
import sys
import numpy as np
import os.path

import experiments as experiments
import csv

import pickle

output_path = "/om/user/xboix/share/robustness/csvs_all/"

################################################################################################
# Read experiment to run
################################################################################################

range_len = 7
knockout_range = np.linspace(0.0, 1.0, num=range_len)
noise_range = np.linspace(0.0, 1.0, num=range_len)
# multiplicative_range = np.arange(0.0, 1.2, 0.2)

noise_idx = [0, 3]
knockout_idx = [1, 2, 4]

for opt in experiments.opt:

    #print(opt.name)


    if os.path.isfile(opt.log_dir_base + opt.name + '/robustness0.pkl'):
        with open(opt.log_dir_base + opt.name + '/robustness0.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        print(opt.name)
        sys.stdout.flush()
        continue

    header = ['model_name', 'cross_validation', 'evaluation_set', 'perturbation_layer', 'perturbation_name', 'perturbation_amount',
              'training_dropout', 'training_weight_decay', 'training_data_augmentation',
              'training_amount_data', 'random_labels', 'scramble_image',
              'multiplier_layer', 'condition', 'performance']

    with open(output_path + opt.name + '_robustness.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(header)

        for k in range(3):

            if os.path.isfile(opt.log_dir_base + opt.name + '/robustness' + str(k) + '.pkl'):
                with open(opt.log_dir_base + opt.name + '/robustness' + str(k) + '.pkl', 'rb') as f:
                    results = pickle.load(f)
            else:
                print(opt.name)
                sys.stdout.flush()
                continue

            # Make sure strict test is in the data
            if not len(np.shape(results)) == 2:
                print("STRICT:" +  opt.name)
                sys.stdout.flush()
                continue

            test_robustness = ['Synaptic Noise', 'Synaptic Knockout',
                               'Activation Knockout', 'Activation Noise', 'Activation Knockout Selected',
                              ]

            eval_sets = ['train set', 'test set']

            for test, idx_test in zip(test_robustness, range(len(test_robustness))):
                if idx_test == 0 or idx_test == 3:
                    range_perturb = noise_range
                else:
                    range_perturb = knockout_range
                #ONLY KNOCKOUT!
                if idx_test != 2 and idx_test != 4 and idx_test != 3:
                    continue

                for eval_set, idx_set in zip(eval_sets, range(len(eval_sets))):
                    for layer in range(opt.dnn.layers+1):
                        for idx_multi, flag_multi in enumerate(['min', 'ave', 'max']):
                            for amount, idx_amount in zip(range_perturb, range(range_len)):

                                ll = []
                                ll.append(opt.dnn.name)                                 # model_name
                                ll.append(k)
                                ll.append(eval_set)                                     # perturbation_set
                                if layer == opt.dnn.layers:
                                    ll.append('all')                                   # perturbation_layer
                                else:
                                    ll.append(layer)                                   # perturbation_layer
                                ll.append(test)                                         # perturbation_name
                                ll.append(amount)                                        # perturbation_amount
                                ll.append(opt.hyper.drop_train)                         # training_dropout
                                ll.append(opt.hyper.weight_decay)                       # training_weight_decay
                                ll.append(opt.hyper.augmentation)                       # training_data_augmentation
                                ll.append(opt.dataset.proportion_training_set)          # training_amount_data
                                ll.append(opt.dataset.random_labels)                    # random_labels
                                ll.append(opt.dataset.scramble_data)                    # scramble_image
                                #for layer_multiplier in range(opt.dnn.layers - 1):
                                #    ll.append(opt.dnn.neuron_multiplier[layer_multiplier])  # multiplier_layerX
                                ll.append(opt.dnn.neuron_multiplier[0])  # multiplier_layerX
                                ll.append(flag_multi)

                                ll.append(results[idx_test][idx_set][layer][idx_amount][idx_multi])    # performance
                                filewriter.writerow(ll)

################################################################################################
print(":)")