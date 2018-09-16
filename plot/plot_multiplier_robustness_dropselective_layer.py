import argparse
import logging
import sys

import functools
import numpy as np
import pandas as pd
import glob
from robustness.plot import plot, mock_cross_validation_if_necessary

from matplotlib import rc
rc('text', usetex=True)
import matplotlib

from matplotlib import pyplot

_logger = logging.getLogger(__name__)

import itertools
import seaborn

def plot_single_data(data, area_under_curve_func=np.trapz):
    data = data[data['condition'] == 'ave']



    nuance_settings = data[['training_dropout', 'random_labels', 'training_weight_decay', 'training_data_augmentation']].drop_duplicates()
    idx = []
    for k in nuance_settings.iterrows():
        if k[1]['training_dropout'] == 0.5:
            if k[1]['training_data_augmentation'] == True:
                idx.append(1)
            else:
                idx.append(0)
        elif k[1]['training_weight_decay'] != 0:
            if k[1]['training_data_augmentation'] == True:
                idx.append(1)
            else:
                idx.append(3)
        elif k[1]['training_data_augmentation'] != 0:
            if k[1]['training_weight_decay'] == True:
                idx.append(1)
            else:
                idx.append(2)
        elif k[1]['random_labels'] == True:
            idx.append(5)
        else:
            idx.append(4)


    nuance_settings = (nuance_setting.to_dict() for _, nuance_setting in nuance_settings.iterrows())
    name_settings = ['Dropout', 'All Regularizers','Data Augment.', 'Weight Decay', 'Not Regularized',  'Random Labels'  ]
    cc = ['skyblue', 'green', 'purple', 'olive', 'darkblue', 'red']

    cc_normal = itertools.cycle(seaborn.light_palette("navy", reverse=False))
    cc_random = itertools.cycle(seaborn.light_palette("red", reverse=False))

    linestyles = ["--","-","--", "--","-","-"]
    ll_width = [3, 4, 3, 3, 4, 4]
    fig, ax = pyplot.subplots()
    ll_factor = ['1/4x', '', '1x', '', '4x']
    for idx_plot, num_data in enumerate(nuance_settings):
        if idx[idx_plot] == 0 or idx[idx_plot] == 2 or idx[idx_plot] == 3 or idx[idx_plot] == 1:
            continue
        if idx[idx_plot] == 4:
            cc = cc_normal
        else:
            cc = cc_random

        for idx_multi, mult in enumerate(sorted(np.unique(data['multiplier_layer']))):
            if mult == 0.5 or mult == 2:
                continue

            col = next(cc)
            layer_data = data[data['training_dropout'] == num_data['training_dropout']].copy()
            layer_data = layer_data[layer_data['multiplier_layer'] == mult]

            layer_data = layer_data[layer_data['training_weight_decay'] == num_data['training_weight_decay']].copy()
            layer_data = layer_data[layer_data['random_labels'] == num_data['random_labels']].copy()
            layer_data = layer_data[layer_data['training_data_augmentation'] == num_data['training_data_augmentation']].copy()
            layer_data_selected = layer_data[layer_data['perturbation_name'] == 'Activation Knockout Selected'].copy()
            layer_data = layer_data[layer_data['perturbation_name'] == 'Activation Knockout'].copy()

            layer_data = compute_areaundercurve(layer_data, area_under_curve_func=area_under_curve_func)
            layer_data_selected = compute_areaundercurve(layer_data_selected, area_under_curve_func=area_under_curve_func)

            layer_data['area_under_curve'] = \
                layer_data['area_under_curve'] - \
                layer_data_selected['area_under_curve']

            layer_data = layer_data.groupby('perturbation_layer')  # group across cross_validations
            means, errs = layer_data['area_under_curve'].mean(), layer_data['area_under_curve'].std()
            means.plot(yerr=errs,  linewidth=ll_width[idx[idx_plot]],
                       label=name_settings[idx[idx_plot]] + ' ' + ll_factor[idx_multi],  linestyle=linestyles[idx[idx_plot]], \
                       color=col, ax=ax)
            col = next(cc)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())




    pyplot.legend(
                  loc='upper left', frameon= True)

    pyplot.xticks(np.arange(4), np.arange(1, 5))

    if np.unique(data['evaluation_set']) == 'train set':
        ax.set_title('CIFAR-10 - Train Set\n Drop in AUC Ablation from Targeting Neurons')
    else:
        ax.set_title('CIFAR-10 - Test Set\n Drop in AUC Ablation from Targeting Neurons ')

    ax.set_xlabel('Layer')
    ax.set_ylabel('AUC Ablation - AUC Targeted Ablation ')
    pyplot.ylim([-0.05, 0.5])


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)

    pyplot.gcf().subplots_adjust(bottom=0.17, top=0.86, left=0.19,right=0.96)
    return fig


def compute_areaundercurve(data, area_under_curve_func=np.trapz):
    prepared_data = []
    for layer in np.unique(data['perturbation_layer']):
        layer_data = data[data['perturbation_layer'] == layer]
        # compute AUC per multiplier and cross-validation
        grouped = layer_data.groupby(['multiplier_layer', 'cross_validation'])

        area_under_curve = grouped.apply(functools.partial(normalized_auc, area_under_curve_func=area_under_curve_func))
        area_under_curve = area_under_curve.reset_index()
        area_under_curve['perturbation_layer'] = layer
        area_under_curve.rename(columns={'multiplier_layer': 'layer_multiplier'}, inplace=True)
        prepared_data.append(area_under_curve)
    return pd.concat(prepared_data)


def normalized_auc(group, area_under_curve_func=np.trapz):
    performance_without_perturbation = group[group['perturbation_amount'] == 0]['performance']
    assert len(performance_without_perturbation) == 1
    performance_without_perturbation = performance_without_perturbation.values[0]

    maximum_auc = max(group['perturbation_amount']) - min(group['perturbation_amount'])

    return pd.Series({
        'area_under_curve': area_under_curve_func(group['performance'], x=group['perturbation_amount'])
                            / performance_without_perturbation / maximum_auc
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running {} with args {}".format(__file__, args))

    data = pd.concat((pd.read_csv(f) for f in glob.glob(args.files[0])))
    data = mock_cross_validation_if_necessary(data)

    data = data[data['training_amount_data'] == 0.95].copy()
    data = data[data['training_amount_data'] == 0.95].copy()
    data = data[data['perturbation_layer'] != "all"].copy()
    data = data[data['perturbation_layer'] != "4"].copy()

    plot(data, plot_single_data,
         plotting_columns=['perturbation_layer','perturbation_name','training_dropout', 'random_labels','training_weight_decay', 'training_data_augmentation', 'random_labels', 'condition', 'perturbation_amount', 'performance', 'cross_validation'] + \
                          [column for column in data.columns if column.startswith('multiplier_layer')],
         results_subdirectory='multiplier_robustness_drop_layer')


if __name__ == '__main__':
    main()
