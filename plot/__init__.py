import logging
import os
import re

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import seaborn

seaborn.set()
seaborn.set_style("whitegrid")
seaborn.set_context("poster")
from matplotlib import pyplot

_logger = logging.getLogger(__name__)

WHOLE_NETWORK = 'all'

def plot(data, plot_single_data, plotting_columns, ignored_key_columns=('performance',), results_subdirectory=None):
    """
    :param pandas.DataFrame data:
    :param callable plot_single_data:
    :param plotting_columns:
    :param ignored_key_columns:
    :param string results_subdirectory:
    :return:
    """
    layer_column = 'perturbation_layer' if 'perturbation_layer' in data.columns else 'layer'
    data = data[data[layer_column] == WHOLE_NETWORK]
    _logger.info("Found {} entries".format(len(data)))
    data = drop_duplicates(data, ignored_key_columns)
    # all possible nuance parameter combinations
    nuance_settings = nuance_combinations(data, plotting_columns)
    for nuance_setting in nuance_settings:
        single_data = data.loc[(data[list(nuance_setting)] == pd.Series(nuance_setting)).all(axis=1)]
        if len(single_data) == 0:  # not every possible setting is realized
            continue
        _logger.debug('Plotting {}'.format(nuance_setting))
        fig = plot_single_data(single_data)
        if fig is None:
            _logger.warning("No result for {}".format(nuance_setting))
            continue
        savepath = '--'.join('{}-{}'.format(_shorten_setting_name(setting_name), setting_value)
                             for setting_name, setting_value in nuance_setting.items()) + '.pdf'
        savepath = os.path.join(get_results_dir(results_subdirectory), savepath)
        _logger.info('Saving to {}'.format(savepath))
        seaborn.color_palette("hls", 8)
        pyplot.savefig(savepath, dpi=1000)
        pyplot.close(fig)


def _shorten_setting_name(setting_name):
    return "".join(map(lambda s: s[0] if not str.isdigit(s) else s,
                       filter(None, re.split('[ _]+|(\d+)', setting_name))))


def drop_duplicates(data, ignored_key_columns, inplace=False):
    key_columns = set(data.columns) - (set(ignored_key_columns) if ignored_key_columns is not None else set())
    num_duplicates = len(data) - len(data.drop_duplicates(subset=key_columns))
    if num_duplicates:
        _logger.warning("Dropping {} duplicates".format(num_duplicates))
        data = data.drop_duplicates(subset=key_columns, inplace=inplace)
    return data


def get_results_dir(subdirectory=None):
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    if subdirectory is not None:
        results_dir = os.path.join(results_dir, subdirectory)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    return results_dir


def nuance_combinations(data, excluded_columns):
    nuance_columns = set(data.columns) - set(excluded_columns)
    column_indices = [list(data.columns.values).index(column) for column in nuance_columns]
    nuance_columns = [col for _, col in sorted(zip(column_indices, nuance_columns))]  # preserve DataFrame ordering
    nuance_settings = data[nuance_columns].drop_duplicates()
    nuance_settings = (nuance_setting.to_dict() for _, nuance_setting in nuance_settings.iterrows())
    return nuance_settings


def mock_cross_validation_if_necessary(data, inplace=False):
    data = data if inplace else data.copy()
    if 'cross_validation' not in data:
        data['cross_validation'] = 1
    return data
