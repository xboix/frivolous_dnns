import argparse
import logging
import sys
import glob

import pandas as pd

from robustness.plot import plot, mock_cross_validation_if_necessary
from robustness.plot.plot_redundancy import information_per_neuron

from matplotlib import pyplot

_logger = logging.getLogger(__name__)


def plot_single_data(data):
    fig, ax = pyplot.subplots()
    layer_data = data.copy()
    layer_data['num_components_per_neuron'] = information_per_neuron(layer_data)
    train_data = layer_data[layer_data['evaluation_set'] == 'train set'].copy().reset_index(drop=True)
    test_data = layer_data[layer_data['evaluation_set'] == 'test set'].copy().reset_index(drop=True)

    # link train and test
    traintest_data = pd.merge(train_data, test_data, on=list(
        set(layer_data.columns) - {'evaluation_set', 'num_components', 'num_components_per_neuron'}))
    assert len(train_data) == len(test_data) == len(traintest_data)
    traintest_data.rename(columns={column: column.replace('_x', '_train').replace('_y', '_test')
                                   for column in traintest_data.columns}, inplace=True)

    # plot means and stds
    traintest_data = traintest_data.groupby('layer')
    traintest_data = traintest_data[['num_components_per_neuron_train', 'num_components_per_neuron_test']]
    means, errs = traintest_data.mean(), traintest_data.std()
    means.plot.scatter(x='num_components_per_neuron_train', y='num_components_per_neuron_test',
                       xerr=errs['num_components_per_neuron_train'], yerr=errs['num_components_per_neuron_test'],
                       c=means.index, ax=ax)
    ax.legend()
    return fig




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running {} with args {}".format(__file__, args))

    data = pd.concat((pd.read_csv(f) for f in glob.glob(args.files[0])))
    del data['performance']
    data = mock_cross_validation_if_necessary(data)

    plot(data, plot_single_data,
         plotting_columns=['layer', 'evaluation_set', 'num_components', 'cross_validation'],
         ignored_key_columns=['num_components'], results_subdirectory='redundancy_traintest')


if __name__ == '__main__':
    main()
