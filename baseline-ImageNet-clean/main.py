import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
import argparse
import sys
import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--network', type=str, required=True)
FLAGS = parser.parse_args()

dataset_path = {
    'xavier': '/Users/xboix/src/baseline-ImageNet/log/',
    'dgx': '/raid/poggio/home/xboix/data/imagenet-tfrecords/',
    'om': '/om/user/xboix/data/ImageNet/'}[FLAGS.host_filesystem]

# path where model is stored
output_path = {
    'xavier': '/Users/xboix/src/baseline-ImageNet/log/',
    'dgx': '/raid/poggio/home/xboix/src/robustness-imagenet/official/resnet/models/',
    'om': '/om/user/xboix/share/robustness/imagenet/'}[FLAGS.host_filesystem]

if FLAGS.network == "all":
    print(os.path)
    sys.stdout.flush()
    from experiments import experiments


def run_test(id):
    from runs import test
    run_opt = experiments.get_experiments(output_path, dataset_path)[id]
    if os.path.exists(run_opt.log_dir_base + run_opt.name):
        test.run(run_opt)
    else:
        print(run_opt.log_dir_base + run_opt.name + " NOT TRAINED OR OUTPUT PATH INVALID")


def run_activations(id):
    from runs import activations
    run_opt = experiments.get_experiments(output_path, dataset_path)[id]
    if os.path.exists(run_opt.log_dir_base + run_opt.name):
        activations.run(run_opt)
    else:
        print(run_opt.log_dir_base + run_opt.name + " NOT TRAINED OR OUTPUT PATH INVALID")


def run_redundancy(id):
    from runs import redundancy
    run_opt = experiments.get_experiments(output_path, dataset_path)[id]
    if os.path.exists(run_opt.log_dir_base + run_opt.name):
        redundancy.run(run_opt)
    else:
        print(run_opt.log_dir_base + run_opt.name + " NOT TRAINED OR OUTPUT PATH INVALID")


def run_robustness(id):
    from runs import robustness
    run_opt = experiments.get_experiments(output_path, dataset_path)[id]
    if os.path.exists(run_opt.log_dir_base + run_opt.name):
        robustness.run(run_opt)
    else:
        print(run_opt.log_dir_base + run_opt.name + " NOT TRAINED OR OUTPUT PATH INVALID")


switcher = {'test': run_test, 'activations': run_activations,
            'redundancy': run_redundancy, 'robustness': run_robustness}

switcher[FLAGS.run](FLAGS.experiment_index)
