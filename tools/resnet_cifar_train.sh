#!/bin/bash
#SBATCH -c2
#SBATCH --array=238
#SBATCH --job-name=cifar_train
#SBATCH --mem=32GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 1-12:00:00
#SBATCH --workdir=/om/user/scasper/workspace/
#SBATCH --qos=cbmm

cd /om/user/scasper/workspace/

# train and export
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/resnet/cifar10_main.py \
--data_dir /om/user/scasper/redundancy/resnet/cifar_data/cifar-10-batches-bin/ \
--model_dir /om/user/scasper/workspace/models/resnet_cifar/ \
--opt_id ${SLURM_ARRAY_TASK_ID}

# --export_dir /om/user/scasper/workspace/models/resnet_cifar/

