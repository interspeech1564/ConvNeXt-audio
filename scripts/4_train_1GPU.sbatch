#!/bin/bash

#SBATCH --job-name=TinyDummy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --ntasks-per-node=1
#####SBATCH --qos=qos_gpu-t3
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=./sbatch_log.out
#SBATCH --error=./sbatch_log.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=10
#SBATCH --account=djl@v100
####SBATCH --account=vby@v100
#SBATCH --hint=nomultithread

PYTHON=/gpfswork/rech/mjp/uzj43um/conda-envs/audio_retrieval/bin/python

# module purge
# module load pytorch-gpu/py3/1.11.0

# echo des commandes
set -x

date

# DATASET_DIR=/gpfsdswork/dataset/AudioSet   # Default first argument.
WORKSPACE_SCRATCH=/gpfsscratch/rech/djl/uzj43um/audioset_tagging   # Default second argument.
DATASPACE=/gpfsstore/rech/djl/uzj43um/audioset

BASEDIR=/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn
SCRIPT=$BASEDIR/pytorch/main.py

export WANDB_MODE=offline
# export TORCH_DISTRIBUTED_DEBUG=INFO

srun $PYTHON -u $SCRIPT train \
    --workspace=$WORKSPACE_SCRATCH \
    --dataspace=$DATASPACE \
    --data_type='full_train' \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='ConvNextTiny' \
    --after_stem_dim 56 \
    --resume_iteration=0 \
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='mixup' \
    --use_wd_scheduler \
    --use_speed_perturb \
    --drop_path_rate=0.4 \
    --batch_size=64 \
    --num_workers=10 \
    --learning_rate=4e-3 \
    --weight_decay=0.05 \
    --early_stop=100 \
    --seed=1978 \
    --cuda


#    --use_speed_perturb \
#    --drop_path_rate=0.2 \
#    --deformable_blocks=1 \

# # Plot statistics
# python3 utils/plot_statistics.py plot \
#     --dataset_dir=$DATASET_DIR \
#     --workspace=$WORKSPACE \
#     --select=1_aug
