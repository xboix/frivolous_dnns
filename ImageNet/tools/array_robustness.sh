#!/bin/bash
#SBATCH --array=23,38
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --job-name=imagenet_array_run
#SBATCH --mem=48GB
#SBATCH -t 7-0:00:00
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
--run=robustness

# if running activations, try 6 cores and 32 GB of mem
