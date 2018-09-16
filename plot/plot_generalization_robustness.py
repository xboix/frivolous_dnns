import argparse
import logging
import sys
import glob

import numpy as np
import pandas as pd

from robustness.plot import plot, mock_cross_validation_if_necessary
from robustness.plot.plot_multiplier_robustness import normalized_auc

from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data, link_test=True):
    color = ['b',  'y',  'c',  'g', 'r' ]
    data = data[data['condition'] == 'ave'].copy()
    fig, ax = pyplot.subplots()
    for  idx_tt, tt in enumerate(np.unique(data['multiplier_layer'])):
        for layer in np.unique(data['perturbation_layer']):
            layer_data = data[data['perturbation_layer'] == layer].copy()
            layer_data = layer_data[layer_data['multiplier_layer'] == tt].copy()

            train_data = layer_data[layer_data['evaluation_set'] == 'train set']
            test_data = layer_data[layer_data['evaluation_set'] == 'test set']

            # compute train robustness for different data regimes
            train_data_regimes = train_data.groupby(['training_amount_data', 'cross_validation'])
            robustness_data = train_data_regimes.apply(normalized_auc)
            robustness_data.reset_index(inplace=True)

            # link train robustness to performance (generalization)
            if link_test:
                generalization_data = test_data[test_data['perturbation_amount'] == 0]
                del generalization_data['evaluation_set']  # there are two evaluation sets in here
            else:
                generalization_data = train_data_regimes.apply(
                    lambda group: pd.Series({'performance': np.mean(group['performance'])})).reset_index()
            robustness_generalization_data = pd.merge(robustness_data, generalization_data)

            # plot
            robustness_generalization_data = robustness_generalization_data.groupby('training_amount_data')[
                'performance', 'area_under_curve']
            means, errs = robustness_generalization_data.mean(), robustness_generalization_data.std()
            means.plot.scatter(x='area_under_curve', y='performance',
                               xerr=errs['area_under_curve'], yerr=errs['performance'],
                               label=tt, ax=ax, color=color[idx_tt])
    ax.legend()
    ax.set_xlabel('robustness = normalized AUC')
    ax.set_ylabel('generalization = test performance')
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running {} with args {}".format(__file__, args))

    data = pd.concat((pd.read_csv(f) for f in glob.glob(args.files[0])))
    data = mock_cross_validation_if_necessary(data)
    data.rename(columns={'perturbation_set': 'evaluation_set'}, inplace=True)

    plot(data, plot_single_data,
         plotting_columns=['condition','training_amount_data', 'perturbation_layer', 'cross_validation', 'evaluation_set',
                           'perturbation_amount', 'performance']
                          ,
         ignored_key_columns=['performance'],
         results_subdirectory='generalization_robustness')


if __name__ == '__main__':
    main()
