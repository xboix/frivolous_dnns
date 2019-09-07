#!/bin/bash
#SBATCH --array=0-11
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --job-name=array_redundancy
#SBATCH --mem=32GB
#SBATCH -t 1:00:00
#SBATCH --workdir=/om/user/scasper/workspace/
#SBATCH --gres=gpu:tesla-k80:4
#SBATCH --qos=cbmm

cd /om/user/scasper/workspace/

hostname

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/baseline-ImageNet-clean/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om \
--network=all \
--run=redundancy

# if running activations, try 6 cores and 32 GB of mem
