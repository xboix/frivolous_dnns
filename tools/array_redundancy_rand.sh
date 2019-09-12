#!/bin/bash

#SBATCH -N 1
#SBATCH -c 2
#SBATCH --array=243-342
#SBATCH --job-name=MLPs
#SBATCH --mem=20GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 2:00:00
#SBATCH -D /om/user/xboix/src/redundancy_dnns/tools/log/
#SBATCH --partition=cbmm

cd /om/user/xboix/src/redundancy_dnns/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/xboix/src/redundancy_dnns/get_redundancy.py ${SLURM_ARRAY_TASK_ID}

# make sure to set the directories and seed in imagenet_networks.py
