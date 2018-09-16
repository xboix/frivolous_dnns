import argparse
import logging
import sys

import numpy as np
import pandas as pd
import glob
from robustness.plot import plot, mock_cross_validation_if_necessary

import matplotlib as mpl

from robustness.plot.plot_multiplier_robustness import normalized_auc

mpl.use('Agg')
mpl.rcParams['ps.useafm'] = True
mpl.rcParams.update({'font.size': 14})
mpl.rc('axes', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('xtick', labelsize=14)

from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data):
    fig, ax = pyplot.subplots()
    layer_multiplier = 'multiplier_layer'
    layer = 'all'
    for multiplier in sorted(np.unique(data[layer_multiplier])):
        layer_data = data[data['perturbation_layer'] == layer].copy()
        layer_data = layer_data[layer_data[layer_multiplier] == multiplier]
        #layer_data = layer_data[layer_data['condition'] == 'ave']

        # compute robustness for layer for different amounts of data
        grouped = layer_data.groupby(['training_amount_data', 'cross_validation'])
        robustness_data = grouped.apply(normalized_auc)
        robustness_data.reset_index(inplace=True)

        # plot means and stds
        robustness_data = robustness_data.groupby('training_amount_data')
        means = robustness_data['area_under_curve'].mean()
        errs = robustness_data['area_under_curve'].std()
        means.plot(yerr=errs, xticks=means.index, label=multiplier, logx=False, ax=ax)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.legend()
    ax.set_xlabel('training_amount_data')
    ax.set_ylabel('robustness = normalized AUC')
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

    plot(data, plot_single_data,
         plotting_columns=['perturbation_layer', 'cross_validation',
                           'perturbation_amount', 'performance', 'training_amount_data']
                          + [column for column in data if column.startswith('multiplier_layer')],
         ignored_key_columns=['performance'],
         results_subdirectory='numdata_robustness')


if __name__ == '__main__':
    main()
