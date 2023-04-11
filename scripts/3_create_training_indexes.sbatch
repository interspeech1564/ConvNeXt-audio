#!/bin/bash

#SBATCH --job-name=ind_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
###SBATCH -C v100-32g
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_gpu-dev
##SBATCH --qos=qos_gpu-t3
#SBATCH --output=./sbatch_log.out
#SBATCH --error=./sbatch_log.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=10
#SBATCH --account=djl@v100
#SBATCH --hint=nomultithread


module purge
PYTHON=/gpfswork/rech/mjp/uzj43um/conda-envs/audio_retrieval/bin/python

# module load pytorch-gpu/py3/1.11.0

# echo des commandes
set -x

date

# DATASET_DIR=/gpfsdswork/dataset/AudioSet   # Default first argument.
WORKSPACE=/gpfsscratch/rech/djl/uzj43um/audioset_tagging   # Default second argument.
DATASPACE=/gpfsstore/rech/djl/uzj43um/audioset

BASEDIR=/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn
SCRIPT=$BASEDIR/utils/create_indexes.py

# # evaluation indexes
# srun $PYTHON -u $SCRIPT create_indexes \
#     --waveforms_hdf5_path=$DATASPACE"/hdf5s/waveforms/eval.h5" \
#     --indexes_hdf5_path=$DATASPACE"/hdf5s/indexes/eval.h5"

# # Balanced training indexes
# srun $PYTHON -u $SCRIPT create_indexes \
#     --waveforms_hdf5_path=$DATASPACE"/hdf5s/waveforms/balanced_train.h5" \
#     --indexes_hdf5_path=$DATASPACE"/hdf5s/indexes/balanced_train.h5"

# Unbalanced trainin# g indexes
# cf launch_3.sh

# # for IDX in {00..10}; do
# #     echo $IDX
# #     srun $PYTHON -u $SCRIPT create_indexes \
# #         --waveforms_hdf5_path=$DATASPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" \
# #         --indexes_hdf5_path=$DATASPACE"/hdf5s/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
# # done

# Combine balanced and unbalanced training indexes to a full training indexes hdf5
srun $PYTHON -u $SCRIPT combine_full_indexes \
    --indexes_hdf5s_dir=$DATASPACE"/hdf5s/indexes" \
    --full_indexes_hdf5_path=$DATASPACE"/hdf5s/indexes/full_train.h5"

# # ============ Blacklist for training (optional) ============
# # Audios in the balck list will not be used in training
# srun $PYTHON -u utils/create_black_list.py dcase2017task4 --workspace=$WORKSPACE
