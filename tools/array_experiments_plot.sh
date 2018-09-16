#!/bin/bash
#SBATCH -n 2
#SBATCH --array=7-9
#SBATCH --job-name=robustness
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om/user/xboix/src/robustness/robustness/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/robustness/robustness/plot_test.py ${SLURM_ARRAY_TASK_ID}


