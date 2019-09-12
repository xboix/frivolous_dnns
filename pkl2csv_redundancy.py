from __future__ import print_function
import experiments
import csv
import os.path
import pickle
import sys
import numpy as np

# from . import experiments

################################################################################################
# Read experiment to run
################################################################################################

for opt in experiments.opt[243:]:

    header = ['model_name', 'evaluation_set', 'cross_validation', 'layer', 'preserved_energy', 'training_dropout',
              'training_weight_decay', 'training_data_augmentation', 'training_amount_data', 'random_labels',
              'scramble_image', 'init_type', 'init_factor', 'random_seed', 'multiplier_layer', 'num_neurons_layer0',
              'num_neurons_layer1', 'num_neurons_layer2', 'num_neurons_layer3', 'num_neurons_layerall',
              'num_components', 'compressability_95', 'selectivity_mean', 'selectivity_std', 'selectivity_gen_mean',
              'selectivity_gen_std', 'not_selective', 'similarity_ave', 'similarity_std', 'performance']

    tmp_name = opt.log_dir_base + opt.name
    #tmp_name = tmp_name[:-7] + '1'
    if not os.path.isfile(tmp_name + '/redundancy0.pkl'):
        print('Couldn\'t find files, skipped:', tmp_name[:-7])
        sys.stdout.flush()
        continue

    tmp_name_red = opt.csv_dir + opt.name
    #tmp_name_red = tmp_name_red[:-7] + '1'
    with open(tmp_name_red + '_redundancy.csv', 'w') as csvfile:

        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(header)

        for cross in range(3):
            if os.path.isfile(tmp_name + '/redundancy' + str(cross) + '.pkl'):
                with open(tmp_name + '/redundancy' + str(cross) + '.pkl', 'rb') as f:
                    results = pickle.load(f)
            else:
                print('Couldn\'t find files, skipped:', tmp_name)
                sys.stdout.flush()
                continue

            if os.path.isfile(tmp_name + '/selectivity' + str(cross) + '.pkl'):
                with open(tmp_name+ '/selectivity' + str(cross) + '.pkl', 'rb') as f:
                    select = pickle.load(f)
                    for i in range(len(select)):
                        for j in range(len(select[i])):
                            select[i][j][np.isnan(select[i][j])] = 0
            else:
                print('Couldn\'t find files, skipped:', tmp_name)
                sys.stdout.flush()
                continue

            if os.path.isfile(tmp_name + '/similarity' + str(cross) + '.pkl'):
                with open(tmp_name + '/similarity' + str(cross) + '.pkl', 'rb') as f:
                    sim = pickle.load(f)
            else:
                print('Couldn\'t find files, skipped:', tmp_name)
                sys.stdout.flush()
                continue

            eval_sets = ['train', 'test']

            energies = (0.95, 0.8)

            for eval_set, idx_set in zip(eval_sets, range(len(eval_sets))):
                for energy, energy_idx in zip(energies, range(len(energies))):
                    for layer in range(opt.dnn.layers):
                        ll = []  # a list which will become a row in the csv
                        ll.append(opt.dnn.name)  # model_name
                        ll.append(eval_set)  # perturbation_set
                        ll.append(cross)
                        if layer == opt.dnn.layers - 1:
                            ll.append('all')  # perturbation_layer
                        else:
                            ll.append(layer)
                        ll.append(energy)  # preserved energy
                        ll.append(opt.hyper.drop_train)  # training_dropout
                        ll.append(opt.hyper.weight_decay)  # training_weight_decay
                        ll.append(opt.hyper.augmentation)  # training_data_augmentation
                        ll.append(opt.dataset.proportion_training_set)  # training_amount_data
                        ll.append(opt.dataset.random_labels)  # random_labels
                        ll.append(opt.dataset.scramble_data)  # scramble_image
                        # for layer_multiplier in range(opt.dnn.layers-1):
                        #    ll.append(opt.dnn.neuron_multiplier[layer_multiplier])  # multiplier_layerX
                        ll.append(opt.init_type)
                        ll.append(opt.hyper.init_factor)
                        ll.append(opt.seed)
                        ll.append(opt.dnn.neuron_multiplier[0])  # multiplier_layerX

                        for layer_multiplier in range(opt.dnn.layers - 1):
                            ll.append(results[4][1][layer_multiplier])  # multiplier_layerX
                        ll.append(np.sum(results[4][1][:]))

                        if layer == opt.dnn.layers - 1:
                            cc = 0
                            for i in range(opt.dnn.layers):
                                cc += results[0][idx_set][i][energy_idx]
                            ll.append(cc)  # modes
                        else:
                            ll.append(results[0][idx_set][layer][energy_idx])  # modes

                        if layer == opt.dnn.layers - 1:
                            cc = 0
                            for i in range(opt.dnn.layers - 1):
                                cc += results[5][idx_set][i]
                            ll.append(cc / (opt.dnn.layers-1))  # modes
                        else:
                            ll.append(results[5][idx_set][layer])  # non_zero

                        if layer == opt.dnn.layers - 1:
                            cc = 0
                            for i in range(opt.dnn.layers - 1):
                                cc += np.mean(select[idx_set][i])
                            ll.append(cc / (opt.dnn.layers - 1))  # modes
                            cc = 0
                            for i in range(opt.dnn.layers - 1):
                                cc += np.std(select[idx_set][i])
                            ll.append(cc / (opt.dnn.layers - 1))  # modes
                            cc = 0
                            for i in range(opt.dnn.layers - 1):
                                cc += np.mean(select[2][i])
                            ll.append(cc / (opt.dnn.layers - 1))  # modes
                            cc = 0
                            for i in range(opt.dnn.layers - 1):
                                cc += np.std(select[2][i])
                            ll.append(cc / (opt.dnn.layers - 1))  # modes
                            cc = 0
                            for i in range(opt.dnn.layers - 1):
                                cc += np.mean(select[idx_set][i] < 0.05)
                            ll.append(cc / (opt.dnn.layers - 1))  # modes

                        else:
                            ll.append(np.mean(select[idx_set][layer]))  # train mean sel
                            ll.append(np.std(select[idx_set][layer]))  # train std sel
                            ll.append(np.mean(select[2][layer]))  # test mean sel
                            ll.append(np.std(select[2][layer]))  # test std sel
                            ll.append(np.mean(select[idx_set][layer] < 0.05))  # mean unselective

                        if layer == opt.dnn.layers - 1:
                            mm = 0
                            ss = 0
                            for i in range(opt.dnn.layers - 1):
                                mm += sim[idx_set][0][i]
                                ss += sim[idx_set][1][i]
                            ll.append(mm / (opt.dnn.layers - 1))  # modes
                            ll.append(ss / (opt.dnn.layers - 1))  # modes
                        else:
                            ll.append(sim[idx_set][0][layer])
                            ll.append(sim[idx_set][1][layer])

                        ll.append(results[3][idx_set][0])  # accuracy

                        filewriter.writerow(ll)

    print('Success:', opt.name)

################################################################################################
print('pkl2csv_redundancy.py')
print(":)")
