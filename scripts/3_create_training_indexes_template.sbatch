#!/bin/bash

#SBATCH --job-name=index
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
###SBATCH -C v100-32g
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_gpu-dev
##SBATCH --qos=qos_gpu-t3
#SBATCH --output=./sbatch_log_%j.out
#SBATCH --error=./sbatch_log_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=10
#SBATCH --account=djl@v100
#SBATCH --hint=nomultithread


PYTHON=/gpfswork/rech/mjp/uzj43um/conda-envs/audio_retrieval/bin/python

module purge
# module load pytorch-gpu/py3/1.11.0

# echo des commandes
set -x

date

# DATASET_DIR=/gpfsdswork/dataset/AudioSet   # Default first argument.
WORKSPACE=/gpfsscratch/rech/djl/uzj43um/audioset_tagging   # Default second argument.
DATASPACE=/gpfsstore/rech/djl/uzj43um/audioset

BASEDIR=/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn
SCRIPT=$BASEDIR/utils/create_indexes.py

IDX=$1

echo $IDX
srun $PYTHON -u $SCRIPT create_indexes \
    --waveforms_hdf5_path=$DATASPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" \
    --indexes_hdf5_path=$DATASPACE"/hdf5s/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"

