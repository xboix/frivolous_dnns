#!/bin/bash
#SBATCH -c1
#SBATCH --array=65
#SBATCH --job-name=get_activations
#SBATCH --mem=12GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 2:00:00
#SBATCH --workdir=/om/user/scasper/workspace/
#SBATCH --qos=cbmm

cd /om/user/scasper/workspace/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/get_activations.py ${SLURM_ARRAY_TASK_ID}

# make sure to set the seed in imagenet_networks.py