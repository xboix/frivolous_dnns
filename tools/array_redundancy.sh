#!/bin/bash
#SBATCH -c1
#SBATCH --array=2-6
#SBATCH --job-name=get_redundancy
#SBATCH --mem=8GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 2:00:00
#SBATCH --chdir=/om/user/scasper/workspace/
#SBATCH --partition=cbmm

cd /om/user/scasper/workspace/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/get_redundancy.py ${SLURM_ARRAY_TASK_ID}

# make sure to set the directories and seed in imagenet_networks.py
