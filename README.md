# Frivolous Units: Wider Networks are not Really That Wide

[Paper](https://arxiv.org/abs/1912.04783) in AAAI 2021 Proceedings

## Authors

* Stephen Casper, scasper@college.harvard.edu

* Xavier Boix, xboix@mit.edu
 
* Vanessa D'Amario

* Ling Guo

* Martin Schrimpf

* Kasper Vinken

* Gabriel Kreiman

## Bibtex:

@article{casper2019frivolous,

  title={Frivolous Units: Wider Networks Are Not Really That Wide},
  
  author={Casper, Stephen and Boix, Xavier and D'Amario, Vanessa and Guo, Ling and Schrimpf, Martin and Vinken, Kasper and Kreiman, Gabriel},
  
  journal={arXiv preprint arXiv:1912.04783},
  
  year={2020}
  
}

## Setup

Network training is implemented in Tensorflow 1.14 (untested with 1.15). Most other dependencies are common such as numpy or scipy. To guarante that things will successfully run, use the docker image from [https://hub.docker.com/r/xboixbosch/tf1.14](https://hub.docker.com/r/xboixbosch/tf1.14).

## Networks

TODO explain how to get our trained networks

## Initializing Experiments 

All networks are associated with an ```Experiment``` object which specifies the network and experimental details and associaes each network with a unique ID number. For all non-ImageNet networks, these experiments are initialized in ```experiments.py``` and for all imagenet networks, these are in ```ImageNet/experiments/experiments.py```. For reference, the ```__main__()``` function inside ```experiments.py``` can also be called to save an ```experiment_lookup.txt``` file to help look up experiments and their IDs. 

After an experiment is configured, its experiment IDs is passed as a command line argument for running experiments for training, prunability, or redundancy. 

## Running Experiments

### For Non-ImageNet Experiments

```create_linear_dataset.py``` and the ```data/``` folder are for creating tfrecords datasets and linear datasets for simple experiments with MLPs. 

```trian.py``` trains networks :)

```get_robustness.py``` along with helper functions from ```perturbations.py``` analyzes network performance under perturbations for prunability analysis. 

```get_activations.py``` extracts activations from networks for redundancy analysis. 

```pkl2csv_robustness.py``` and ```pkl2csv_redundancy.py``` process robustness and activations data and store them in csvs which can be later accessed for plotting. 

### For ImageNet Experiments

Analogous files to the above exist in ```ImageNet/runs/```.

## Plotting

The six notebooks in ```ipy_notebooks``` reproduce the plots from the paper. 
