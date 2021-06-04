# Frivolous Units: Wider Networks are not Really That Wide

[Paper](https://arxiv.org/abs/1912.04783) in AAAI 2021 Proceedings

[![Quick video summary:]()](https://www.youtube.com/watch?v=1uDo-6UuW2o)

[![Quick video summary](https://img.youtube.com/vi/1uDo-6UuW2o/0.jpg)](https://www.youtube.com/watch?v=1uDo-6UuW2o)


## Authors

* Stephen Casper, scasper@college.harvard.edu

* Xavier Boix, xboix@mit.edu
 
* Vanessa D'Amario

* Ling Guo

* Martin Schrimpf

* Kasper Vinken

* Gabriel Kreiman

You are welcome to email us. 

## Bibtex:

@inproceedings{casper2021frivolous,
  title={Frivolous Units: Wider Networks Are Not Really That Wide},
  author={Casper, Stephen and Boix, Xavier and D'Amario, Vanessa and Guo, Ling and Schrimpf, Martin and Vinken, Kasper and Kreiman, Gabriel},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35}
  year={2021}
}

## Setup

Network training is implemented in Tensorflow 1.14 (untested with 1.15). To guarante that things will successfully run, use the docker image from [https://hub.docker.com/r/xboixbosch/tf1.14](https://hub.docker.com/r/xboixbosch/tf1.14).

## Initializing Experiments 

All networks are associated with an ```Experiment``` object which specifies the network and experimental details and associaes each network with a unique ID number. For all non-ImageNet networks, these experiments are initialized in ```experiments.py``` and for all imagenet networks, these are in ```ImageNet/experiments/experiments.py```. For reference, the ```__main__()``` function inside ```experiments.py``` can also be called to save an ```experiment_lookup.txt``` file to help look up experiments and their IDs. 

After an experiment is configured, its experiment IDs is passed as a command line argument for running experiments for training, prunability, or redundancy. 

Make sure to adjust the paths in ```experiments.py``` that point to the datasets and results files.

## Running Experiments

### For Non-ImageNet Experiments

```create_linear_dataset.py``` (change the constant DIMENSIONALITY in the code to generate the desired dataset) and the ```data/``` folder are for creating tfrecords datasets and linear datasets for simple experiments with MLPs. 

```train.py``` trains networks.

```get_activations.py``` extracts activations from networks for redundancy analysis. 

```get_redundancy.py``` analyzes redundancy among units. 

```get_robustness.py``` along with helper functions from ```perturbations.py``` analyzes network performance under perturbations for prunability analysis. 

```pkl2csv_robustness.py``` and ```pkl2csv_redundancy.py``` process robustness and activations data and store them in csvs which can be later accessed for plotting. 

### For ImageNet Experiments

Analogous files to the above exist in ```ImageNet/runs/```.

Pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1jce491LXys18_za-VOFnuJlG2cpeMjiO?usp=sharing) to the network training directory. 

## Plotting

The six notebooks in ```ipy_notebooks``` reproduce the plots from the paper. 

## Running Demo Experiments

This demo will train and run experiments for 5 standard AlexNet architectures with Glorot initialization on CIFAR-10. The first step is to download the data into a data directory. 

To get the CIFAR-10 dataset, navigate to your preferred directory, and type ```wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz``` followed by ```tar -zxvf cifar-10-python.tar.gz```, and rename the folder with ```mv cifar-10-batches-py cifar10```.

Second, in lines 4, 5, and 7 of ```experiments.py```, set ```dataset_stem```, ```log_dir_stem```, and ```csv_dir_stem``` to your data directory, where you want models saved, and where you want csvs saved respectively.

Then inside the docker container, run

```
for experiment_id in {2..6}
do
   python train.py experiment_id
   python get_activations.py experiment_id
   python get_redundancy.py experiment_id
   python get_robustness.py experiment_id
done

python pkl2csv_redundancy.py
python pkl2csv_robustness.py
```

Finally, to plot the accuracy, prunability, and redundancy of these networks as a function of width factor, run the notebook ```plot_accuracy.ipynb```.
