#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --array=243-292
#SBATCH --job-name=mlps
#SBATCH --mem=30GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 1-1:00:00
#SBATCH -D /om/user/xboix/src/redundancy_dnns/tools/log/
#SBATCH --partition=cbmm

cd /om/user/xboix/src/redundancy_dnns/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/xboix/src/redundancy_dnns/train.py ${SLURM_ARRAY_TASK_ID}

# make sure to set the seed in imagenet_networks.py

