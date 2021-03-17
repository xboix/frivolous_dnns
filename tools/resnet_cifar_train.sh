#!/bin/bash
#SBATCH -c1
#SBATCH --array=227
#SBATCH --job-name=cifar_train
#SBATCH --mem=12GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 1-12:00:00
#SBATCH --chdir=/om/user/scasper/workspace/
#SBATCH --partition=cbmm

cd /om/user/scasper/workspace/

# train and export
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/resnet/cifar10_main.py \
--data_dir /om/user/scasper/redundancy/resnet/cifar_data/cifar-10-batches-bin/ \
--model_dir /om/user/scasper/workspace/models/resnet_cifar/ \
--opt_id ${SLURM_ARRAY_TASK_ID}

# --export_dir /om/user/scasper/workspace/models/resnet_cifar/

