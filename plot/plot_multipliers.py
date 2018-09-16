import argparse
import logging
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot
import glob
import seaborn
from matplotlib import rc
rc('text', usetex=True)
import itertools

from robustness.plot import plot

_logger = logging.getLogger(__name__)
DATASET = "ImageNet"

def plot_single_data(data):

    fig, ax = pyplot.subplots()
    cc_normal = itertools.cycle(seaborn.light_palette("navy", reverse=False))
    ll = ['1/4x', '1/2x', '1x', '2x', '4x']

    cc_random = itertools.cycle(seaborn.light_palette("red", reverse=False))
    for ff in np.unique(data['random_labels']):
        m_data = data[data['random_labels'] == ff].copy()

        for idx_mult, multiplier in enumerate(sorted(np.unique(m_data['multiplier_layer']))):
            multiplier_data = m_data[m_data['multiplier_layer'] == multiplier].copy()
            multiplier_data = multiplier_data[multiplier_data['condition'] == 'ave']
            if np.unique(data['perturbation_name']) == "Activation Noise":
                multiplier_data['perturbation_amount'] = multiplier_data['perturbation_amount']
            else:
                multiplier_data['perturbation_amount'] = 100*multiplier_data['perturbation_amount']

            mm = multiplier_data
            #
            # plot
            multiplier_data = multiplier_data.groupby('perturbation_amount')
            means, errs = 100*multiplier_data['performance'].mean(), 100*multiplier_data['performance'].std()
            if ff==True:
                cc = cc_random
            else:
                cc = cc_normal

            means.plot(yerr=errs,  ax=ax, color = next(cc), label=ll[idx_mult])
    mm['performance'] = 0.1
    mm = mm.groupby('perturbation_amount')

    means, errs = 100 * mm['performance'].mean(), 100 * mm['performance'].std()
    means.plot(yerr=errs, ax=ax, color='black', linestyle=':', label='Random Guess')
    if np.unique(data['random_labels']):
        if not np.unique(data['perturbation_name']) == "Activation Noise":
            ax.set_title(DATASET + "\n" + r"Accuracy after Ablation - $\textbf{Random labels}$")
        else:
            ax.set_title(DATASET + "\n" + r"Accuracy after Adding Noise - $\textbf{Random labels}$")

    else:
        if not np.unique(data['perturbation_name']) == "Activation Noise":
            ax.set_title(DATASET + "\n" + r"Accuracy after Ablation - $\textbf{Real labels}$")
        else:
            ax.set_title(DATASET + "\n" + r"Accuracy after Adding Noise - $\textbf{Real labels}$")


    handles, labels = pyplot.gca().get_legend_handles_labels()

    ax.set_ylabel('Accuracy (\%)')

    if np.unique(data['perturbation_name']) == "Activation Noise":
        ax.set_xlabel(r"$\sigma$(noise) / $\sigma$(activations)")
        ax.legend(title='Model Size Factor', frameon=True)

    else:
        ax.set_xlabel('\% Ablated Neurons')
        ax.legend(title='Model Size Factor',  frameon=True)

    pyplot.ylim([0, 100])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)

    pyplot.gcf().subplots_adjust(bottom=0.15, top=0.87, left=0.15,right=0.95)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running {} with args {}".format(__file__, args))

    data = pd.concat((pd.read_csv(f) for f in glob.glob(args.files[0])))

    data = data[data['training_amount_data'] == 0.95]

    data = data[data['training_dropout'] == 1.00].copy()
    data = data[data['training_weight_decay'] == 0.00 ].copy()
    #data = data[data['training_data_augmentation'] == False].copy()
    data = data[data['perturbation_layer'] == "all"].copy()


    plot(data, plot_single_data,
         plotting_columns=[ 'condition','perturbation_amount', 'performance', 'cross_validation']
                          + [column for column in data if column.startswith('multiplier_layer')],
         results_subdirectory='layer_multipliers')


if __name__ == '__main__':
    main()
