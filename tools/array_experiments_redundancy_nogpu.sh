#!/bin/bash
#SBATCH -n 1
#SBATCH --array=66-70
#SBATCH --job-name=redundancy
#SBATCH --mem=12GB
#SBATCH -t 10:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om/user/xboix/src/robustness/robustness/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/xboix/src/robustness/robustness/redundancy_nogpu.py ${SLURM_ARRAY_TASK_ID}


