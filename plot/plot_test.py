from __future__ import print_function

import os
import pickle
import sys

import experiments
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['ps.useafm'] = True
mpl.rcParams.update({'font.size': 14})
mpl.rc('axes', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('xtick', labelsize=14)

#import seaborn as sns
#sns.set()
import numpy as np

import matplotlib.pyplot as plt

range_len = 7
knockout_range = np.linspace(0.0, 1.0, num=range_len)
noise_range = np.logspace(0.0, 0.5, num=range_len)

noise_idx = [0, 3]
knockout_idx = [1, 2, 4]
#multiplicative_idx = [5, 6]


def main():
    ################################################################################################
    # Read experiment to run
    ################################################################################################

    ID = int(sys.argv[1:][0])

    opt = experiments.opt[ID]

    # Skip execution if instructed in experiment
    if opt.skip:
        print("SKIP")
        quit()

    print(opt.name)
    ################################################################################################

    with open(opt.log_dir_base + opt.name + '/robustness.pkl', 'rb') as f:
        results = pickle.load(f)

    print(results)

    save_dir = opt.log_dir_base + opt.name

    for i in knockout_idx:
        _plot(opt, results[i][0], knockout_range, str(i) + '-Train', save_dir=save_dir)
        _plot(opt, results[i][1], knockout_range, str(i) + '-Validation', save_dir=save_dir)

    for i in noise_idx:
        _plot(opt, results[i][0], noise_range, str(i) + '-Train', save_dir=save_dir)
        _plot(opt, results[i][1], noise_range, str(i) + '-Validation', save_dir=save_dir)

    '''
    for i in multiplicative_idx:
        _plot(opt, results[i][0], noise_range, str(i) + '-Train', save_dir=save_dir)
        _plot(opt, results[i][1], noise_range, str(i) + '-Validation', save_dir=save_dir)
    '''

    print("Done :)")


def _plot(opt, res, res_y, title, save_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title)
    #_set_color_palette(opt.dnn.layers)
    for layer in range(opt.dnn.layers + 1):
        ax.plot(res_y, np.squeeze(res[layer][:]),
                label="Layer" + str(layer))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Neural Death')
    #ax.set_xlim(0, 1)

    fig.tight_layout()
    ax.legend(loc="lower left")
    fig.show()

    savepath = os.path.join(save_dir, 'plot_' + title + '.pdf')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    print("Saving to {}".format(savepath))
    plt.savefig(savepath, format='pdf', dpi=1000)
    plt.close()

'''
def _set_color_palette(num_colors):
    palette = sns.color_palette('deep', min(6, num_colors))  # default palette has only 6 colors
    if len(palette) < num_colors:
        palette += sns.color_palette("husl", num_colors)[:num_colors - len(palette)]
    sns.set_palette(palette)
'''

if __name__ == '__main__':
    main()
