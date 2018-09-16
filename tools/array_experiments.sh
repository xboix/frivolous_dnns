#!/bin/bash
#SBATCH -n 2
#SBATCH --array=2-3
#SBATCH --job-name=robustness
#SBATCH --mem=8GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 15:00:00
#SBATCH --workdir=/om/user/xboix/share/robustness/log/
#SBATCH --qos=cbmm



/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'
cd /om/user/xboix/src/robustness/robustness/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/xboix/src/robustness/robustness/main.py ${SLURM_ARRAY_TASK_ID}


#152