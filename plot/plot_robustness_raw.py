import argparse
import logging
import sys

import pandas as pd

import glob

from robustness.plot import plot

from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data):
    fig, ax = pyplot.subplots()
    data = data[data['condition'] == 'ave'].copy()
    data = data[~(data['perturbation_layer'] == '4')].copy()

    means = data.groupby(['perturbation_layer', 'perturbation_amount'])['performance'].mean().unstack().T
    errs = data.groupby(['perturbation_layer', 'perturbation_amount'])['performance'].std().unstack().T
    means.plot(yerr=errs, label='perturbation_layer', ax=ax)
    ax.set_xlabel('perturbation amount')
    ax.set_ylabel('performance')
    pyplot.ylim([0, 1])
    ax.legend(title='layer')
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running {} with args {}".format(__file__, args))

    data = pd.concat((pd.read_csv(f) for f in glob.glob(args.files[0])))
    plot(data, plot_single_data,
         plotting_columns=['condition','perturbation_layer', 'perturbation_amount', 'performance', 'cross_validation'],
         results_subdirectory='robustness_raw')


if __name__ == '__main__':
    main()
