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

range_len = 7
perturbation_range = np.linspace(0.0, 1.0, num=range_len, endpoint=True)

for opt in run_opts:

    header = ['model_name', 'evaluation_set', 'cross', 'perturbation_layer', 'size_factor', 'batch_size',
              'perturbation_type', 'perturbation_amount', 'unchanged labels']

    if not os.path.isfile(opt.results_dir + opt.name + '/robustness0.pkl'):
        print('Couldn\'t find files, skipped:', opt.name)
        sys.stdout.flush()
        continue

    with open(opt.csv_dir + opt.name + '_robustness.csv', 'w') as csvfile:

        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(header)

        for cross in range(3):

            if os.path.isfile(opt.results_dir + opt.name + '/robustness' + str(cross) + '.pkl'):
                with open(opt.results_dir + opt.name + '/robustness' + str(cross) + '.pkl', 'rb') as f:
                    robustness_results = pickle.load(f)
            else:
                print('Couldn\'t find files, skipped:', opt.name)
                sys.stdout.flush()
                continue

            num_layers = opt.dnn.layers  # should be 5 for resnet, 16 for inception v3

            for ptype in [2, 3, 4]:  # random ablaiton, noise, targeted perturbation
                for p_idx in range(range_len):
                    unchanged_mean_layers = []

                    for layer in range(num_layers):  # the +1 is for the all portion

                        unchanged_mean_layers.append(robustness_results[ptype, layer, p_idx])

                    for layer in range(num_layers + 1):  # the +1 is for the all portion

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
                        ll.append(ptype)  # perturbation type
                        ll.append(perturbation_range[p_idx])  # amount of perturbation

                        if layer < num_layers:
                            ll.append(unchanged_mean_layers[layer])  # unchanged labels for layer perturbation

                        else:
                            ll.append(sum(unchanged_mean_layers)/num_layers)  # unchanged labels for network perturbation

                        filewriter.writerow(ll)

    print('Success:', opt.name)

################################################################################################
print('pkl2csv_robustness.py')
print(":)")
