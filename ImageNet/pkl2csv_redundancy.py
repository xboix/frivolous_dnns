from __future__ import print_function
import csv
import os.path
import pickle
import sys
import numpy as np
from experiments import experiments

output_path = '/om/user/xboix/share/robustness/imagenet/'
dataset_path = '/om/user/xboix/data/ImageNet/'
run_opts = experiments.get_experiments(output_path, dataset_path)

for opt in run_opts:

    header = ['model_name', 'evaluation_set', 'cross', 'layer', 'size_factor', 'batch_size', 'similarity_mean',
              'similarity_std', 'compressability_95', 'selectivity_mean', 'selectivity_std', 'nonselective',
              'top_1_acc', 'top_5_acc']

    if not os.path.isfile(opt.results_dir + opt.name + '/selectivity0.pkl'):
        print('Couldn\'t find files, skipped:', opt.name)
        sys.stdout.flush()
        continue

    with open(opt.csv_dir + opt.name + '_redundancy' + str(cross) + '.csv', 'w') as csvfile:

        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(header)

        for cross in range(3):

            if os.path.isfile(opt.results_dir + opt.name + '/accuracy' + str(cross) + '.pkl'):
                with open(opt.results_dir + opt.name + '/accuracy' + str(cross) + '.pkl', 'rb') as f:
                    acc_results = pickle.load(f)
            else:
                print('Couldn\'t find files, skipped:', opt.name)
                sys.stdout.flush()
                continue

            if os.path.isfile(opt.results_dir + opt.name + '/similarity' + str(cross) + '.pkl'):
                with open(opt.results_dir + opt.name + '/similarity' + str(cross) + '.pkl', 'rb') as f:
                    similarity_results = pickle.load(f)
            else:
                print('Couldn\'t find files, skipped:', opt.name)
                sys.stdout.flush()
                continue

            if os.path.isfile(opt.results_dir + opt.name + '/compressability' + str(cross) + '.pkl'):
                with open(opt.results_dir + opt.name + '/compressability' + str(cross) + '.pkl', 'rb') as f:
                    compressability_results = pickle.load(f)
            else:
                print('Couldn\'t find files, skipped:', opt.name)
                sys.stdout.flush()
                continue

            if os.path.isfile(opt.results_dir + opt.name + '/selectivity' + str(cross) + '.pkl'):
                with open(opt.results_dir + opt.name + '/selectivity' + str(cross) + '.pkl', 'rb') as f:
                    selectivity_results = pickle.load(f)
                    for i in range(len(selectivity_results)):
                        selectivity_results[i][np.isnan(selectivity_results[i])] = 0

            num_layers = opt.dnn.layers  # should be 5

            sim_mean_layers = []
            sim_std_layers = []
            compressability_layers = []
            selectivity_mean_layers = []
            selectivity_std_layers = []
            nonselective_layers = []

            for layer in range(num_layers):
                sim_mean_layers.append(similarity_results[layer][0])
                sim_std_layers.append(similarity_results[layer][1])
                compressability_layers.append(compressability_results[layer])
                selectivity_mean_layers.append(np.mean(selectivity_results[layer]))
                selectivity_std_layers.append(np.std(selectivity_results[layer]))
                nonselective_layers.append(np.mean(selectivity_results[layer] < 0.05))

            for layer in range(num_layers+1):  # plus one is for all layers

                ll = []  # a list which will become a row in the csv
                ll.append(opt.name)  # model_name
                ll.append('validation')  # evaluation set
                ll.append(cross)
                if layer < num_layers:  # if a normal layer and not the all layer column
                    ll.append(layer)  # layer
                else:  # if this is the final iteration and for the all layer column
                    ll.append('all')  # layer

                ll.append(opt.dnn.factor)  # size _factor
                ll.append(opt.hyper.batch_size)  # batch_size

                if layer < num_layers:  # for a normal layer and not the all layer column
                    ll.append(sim_mean_layers[layer])
                    ll.append(sim_std_layers[layer])
                    ll.append(compressability_layers[layer])
                    ll.append(selectivity_mean_layers[layer])
                    ll.append(selectivity_std_layers[layer])
                    ll.append(nonselective_layers[layer])

                else:  # if this is the final iteration and for the all layer column
                    ll.append(sum(sim_mean_layers)/num_layers)
                    ll.append(sum(sim_std_layers)/num_layers)
                    ll.append(sum(compressability_layers)/num_layers)
                    ll.append(sum(selectivity_mean_layers)/num_layers)
                    ll.append(sum(selectivity_std_layers)/num_layers)
                    ll.append(sum(nonselective_layers)/num_layers)

                ll.append(acc_results[0])  # top 1 acc
                ll.append(acc_results[1])  # top 5 acc

                filewriter.writerow(ll)

    print('Success:', opt.name)

################################################################################################
print('pkl2csv_redundancy.py')
print(":)")
