import argparse
import functools
import logging
import sys
import glob
import numpy as np
import pandas as pd

from robustness.plot import plot, drop_duplicates, mock_cross_validation_if_necessary
from robustness.plot.plot_multiplier_robustness import normalized_auc
from robustness.plot.plot_redundancy import information_per_neuron, add_whole_network_redundancy

import seaborn
from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data, area_under_curve_func=np.trapz):
    plot_data = []
    layer = 'all'
    #for layer in np.unique(data['perturbation_layer']):
    layer_data = data[data['perturbation_layer'] == 'all'].copy()
    layer_data = layer_data[layer_data['condition'] == 'ave'].copy()

    layer_data['num_components_per_neuron'] = information_per_neuron(
        layer_data.rename(columns={'perturbation_layer': 'layer'}))

    # group
    def group_apply(group):
        assert len(np.unique(group['multiplier_layer'])) == 1
        auc_group = group.groupby('cross_validation')
        area_under_curve = auc_group.apply(functools.partial(normalized_auc,
                                                             area_under_curve_func=area_under_curve_func))
        auc_mean = area_under_curve.mean()
        assert len(auc_mean) == 1
        return pd.Series({
            'information_mean': group['num_components_per_neuron'].mean(),
            'robustness_mean': auc_mean.values[0]
        })

    grouped = layer_data.groupby(['multiplier_layer', 'cross_validation'])
    layer_data = grouped.apply(group_apply)
    layer_data.reset_index(inplace=True)
    layer_data['layer'] = layer
    plot_data.append(layer_data)

    fig, _ = pyplot.subplots()
    plot_data = pd.concat(plot_data)
    ax = seaborn.lmplot(data=plot_data, x='robustness_mean', y='information_mean', hue='layer',
                        fit_reg=False, ci=None, line_kws={'alpha': 0.3})
    ax.set_axis_labels('redundancy', 'robustness = normalized area under curve')
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robustness_files', type=str, nargs='+', required=True)
    parser.add_argument('--redundancy_files', type=str, nargs='+', required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running {} with args {}".format(__file__, args))

    # read data
    robustness_data = pd.concat((pd.read_csv(f) for f in glob.glob(args.robustness_files[0])))
    robustness_data = mock_cross_validation_if_necessary(robustness_data)
    redundancy_data = pd.concat((pd.read_csv(f) for f in glob.glob(args.redundancy_files[0])))
    del redundancy_data['performance']  # avoid joining on this
    redundancy_data = add_whole_network_redundancy(redundancy_data)  # compute PCA for whole network
    redundancy_data.rename(columns={'layer': 'perturbation_layer'}, inplace=True)

    # drop duplicates and merge
    # see warning above https://pandas.pydata.org/pandas-docs/stable/merging.html#checking-for-duplicate-keys
    drop_duplicates(robustness_data, ignored_key_columns=['performance'], inplace=True)
    drop_duplicates(redundancy_data, ignored_key_columns=['num_components'], inplace=True)
    data = pd.merge(robustness_data, redundancy_data, on=None, left_index=False, right_index=False)

    # plot
    plotting_columns = ['condition','non_zero','perturbation_layer', 'perturbation_amount', 'performance'] \
                       + [column for column in robustness_data if column.startswith('multiplier_layer')] \
                       + ['num_components', 'cross_validation',  'selectivity_mean','selectivity_std','selectivity_gen_mean','selectivity_gen_std'] \
                       + [column for column in redundancy_data if column.startswith('num_neurons_layer')]
    plot(data, plot_single_data, plotting_columns=plotting_columns, results_subdirectory='robustness_vs_redundancy')


if __name__ == '__main__':
    main()
