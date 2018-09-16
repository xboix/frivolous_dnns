import os

logging_format = '[%(asctime)s] %(levelname)s:%(name)s:%(message)s'

_results_dir = 'output'


def results_filepath(dataset, model, perturbation, dropout, weight_decay):
    return os.path.join(_results_dir, 'data_{}-model_{}_dropout_{}_weightdecay_{}-perturbation_{}'.format(
        dataset, model, dropout, weight_decay, perturbation))
