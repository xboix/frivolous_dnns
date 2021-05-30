#!/bin/bash
#SBATCH --array=12-18
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --job-name=array_activations
#SBATCH --mem=48GB
#SBATCH -t 1-0:00:00
#SBATCH --chdir=/om/user/scasper/workspace/
#SBATCH --gres=gpu:tesla-k80:4
#SBATCH --partition=cbmm

cd /om/user/scasper/workspace/

hostname

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/ImageNet/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om \
--network=all \
--run=activations
