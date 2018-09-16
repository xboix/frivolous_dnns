#!/bin/bash
#SBATCH -n 2
#SBATCH --array=66-70
#SBATCH --job-name=robustness
#SBATCH --mem=12GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 10:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

hostname

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'
cd /om/user/xboix/src/robustness/robustness/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/xboix/src/robustness/robustness/test_robustness.py ${SLURM_ARRAY_TASK_ID}


