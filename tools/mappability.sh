#!/bin/bash
#SBATCH -c6
#SBATCH --job-name=mappability
#SBATCH --mem=72G
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 3-0:00:00
#SBATCH --workdir=/om/user/scasper/workspace/
#SBATCH --qos=cbmm

cd /om/user/scasper/workspace/
singularity exec -B /om:/om --nv /om/user/scasper/singularity/xboix-tensorflow.simg \
python /om/user/scasper/redundancy/get_mappability.py ${SLURM_ARRAY_TASK_ID}
