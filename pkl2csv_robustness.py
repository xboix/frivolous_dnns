from __future__ import print_function
import sys
import numpy as np
import os.path
import experiments as experiments
import csv
import pickle


################################################################################################
# Read experiment to run
################################################################################################

range_len = 7
knockout_range = np.linspace(0.0, 1.0, num=range_len, endpoint=True)
noise_range = np.linspace(0.0, 1.0, num=range_len, endpoint=True)
# multiplicative_range = np.arange(0.0, 1.2, 0.2)
noise_idx = [0, 3]
knockout_idx = [1, 2, 4]

for opt in experiments.opt[2:7]:

    output_path = opt.csv_dir

    if os.path.isfile(opt.log_dir_base + opt.name + '/robustness0.pkl'):
        with open(opt.log_dir_base + opt.name + '/robustness0.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        print('Couldn\'t find files, skipped:', opt.name)
        sys.stdout.flush()
        continue

    header = ['model_name', 'cross_validation', 'evaluation_set', 'perturbation_layer', 'perturbation_name',
              'perturbation_amount', 'training_dropout', 'training_weight_decay', 'training_data_augmentation',
              'init_type', 'training_amount_data', 'random_labels', 'scramble_image',
              'multiplier_layer',  # 'condition',
              'performance']

    with open(output_path + opt.name + '_robustness.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(header)

        for k in range(3):

            if os.path.isfile(opt.log_dir_base + opt.name + '/robustness' + str(k) + '.pkl'):
                with open(opt.log_dir_base + opt.name + '/robustness' + str(k) + '.pkl', 'rb') as f:
                    results = pickle.load(f)
            else:
                print('Couldn\'t find files, skipped:', opt.name)
                sys.stdout.flush()
                continue

            # Make sure strict test is in the data
            if not len(np.shape(results)) == 2:
                print("STRICT:" + opt.name)
                sys.stdout.flush()
                continue

            ptypes = ['Synaptic Noise', 'Synaptic Knockout', 'Activation Knockout', 'Activation Noise',
                      'Activation Knockout Selected']

            eval_sets = ['train', 'test']

            for ptype, idx_ptype in zip(ptypes, range(len(ptypes))):
                if idx_ptype == 0 or idx_ptype == 3:
                    range_perturb = noise_range
                else:
                    range_perturb = knockout_range

                if idx_ptype not in [2]:  # only knockout
                    continue

                for eval_set, idx_set in zip(eval_sets, range(len(eval_sets))):
                    for layer in range(opt.dnn.layers + 1):
                        # for idx_multi, flag_multi in enumerate(['min', 'ave', 'max']):
                        for amount, idx_amount in zip(range_perturb, range(range_len)):

                            ll = []
                            ll.append(opt.dnn.name)  # model_name
                            ll.append(k)
                            ll.append(eval_set)  # perturbation_set
                            if layer == opt.dnn.layers:
                                ll.append('all')  # perturbation_layer
                            else:
                                ll.append(layer)  # perturbation_layer
                            ll.append(ptype)  # perturbation_name
                            ll.append(amount)  # perturbation_amount
                            ll.append(opt.hyper.drop_train)  # training_dropout
                            ll.append(opt.hyper.weight_decay)  # training_weight_decay
                            ll.append(opt.hyper.augmentation)  # training_data_augmentation
                            ll.append(opt.init_type)
                            ll.append(opt.dataset.proportion_training_set)  # training_amount_data
                            ll.append(opt.dataset.random_labels)  # random_labels
                            ll.append(opt.dataset.scramble_data)  # scramble_image
                            # for layer_multiplier in range(opt.dnn.layers - 1):
                            #    ll.append(opt.dnn.neuron_multiplier[layer_multiplier])  # multiplier_layerX
                            ll.append(opt.dnn.neuron_multiplier[0])  # multiplier_layerX
                            # ll.append(flag_multi)

                            ll.append(results[idx_ptype][idx_set][layer][idx_amount][0])  # [idx_multi])  # performance
                            filewriter.writerow(ll)

    print('Success:', opt.name)

################################################################################################
print('pkl2csv_robustness.py')
print(":)")
