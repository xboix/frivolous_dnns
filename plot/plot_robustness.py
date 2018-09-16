import argparse
import logging
import sys
import numpy as np
import pandas as pd
import glob
from robustness.plot import plot, mock_cross_validation_if_necessary
from robustness.plot.plot_multiplier_robustness import normalized_auc

from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data):
    data = data[data['condition'] == 'ave'].copy()
    data = data[~(data['perturbation_layer'] == '4')].copy()
    data = data[~(data['perturbation_layer'] == 'all')].copy()
    fig, ax = pyplot.subplots()
    for multiplier in sorted(np.unique(data['random_labels'])):
        layer_data = data[data['random_labels'] == multiplier].copy()
        grouped = layer_data.groupby(['perturbation_layer', 'cross_validation'])
        area_under_curve = grouped.apply(normalized_auc)
        means = area_under_curve.groupby('perturbation_layer')['area_under_curve'].mean()
        errs = area_under_curve.groupby('perturbation_layer')['area_under_curve'].std()
        means.plot(yerr=errs, ax=ax, label=str(multiplier))
    ax.set_xlabel('layer')
    ax.legend(title='Random Labels')
    ax.set_ylabel('robustness = normalized area under curve')
    pyplot.ylim([0, 1])
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
         plotting_columns=['condition', 'random_labels', 'perturbation_layer', 'perturbation_amount', 'performance', 'cross_validation'],
         results_subdirectory='robustness')


if __name__ == '__main__':
    main()
