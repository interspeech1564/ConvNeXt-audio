#!/bin/bash

#SBATCH --job-name=pann_pack
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH -C v100-32g
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_gpu-dev
##SBATCH --qos=qos_gpu-t3
#SBATCH --output=./sbatch_log.out
#SBATCH --error=./sbatch_log.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --account=djl@v100
#SBATCH --hint=nomultithread

module purge

PYTHON=/gpfswork/rech/mjp/uzj43um/conda-envs/audio_retrieval/bin/python

# module load pytorch-gpu/py3/1.11.0

# echo des commandes
set -x

date

# DSDIR : DATASET_DIR=/gpfsdswork/dataset/AudioSet   # Default first argument.
DATASPACE=/gpfsstore/rech/djl/uzj43um/audioset
WORKSPACE=/gpfsscratch/rech/djl/uzj43um/audioset_tagging   # Default second argument.

# DATASET_DIR=${1:-"./datasets/audioset201906"}   # Default first argument.
# WORKSPACE=${2:-"./workspaces/audioset_tagging"}   # Default second argument.

# Pack evaluation waveforms to a single hdf5 file
BASEDIR=/gpfswork/rech/djl/uzj43um/audio_retrieval/audioset_tagging_cnn
SCRIPT=$BASEDIR/utils/dataset.py

# srun $PYTHON -u $SCRIPT pack_waveforms_to_hdf5 \
#     --csv_path=$DATASPACE"/data/eval_segments_19393.csv" \
#     --audios_dir=$DATASPACE"/data/eval_segments/audio" \
#     --waveforms_hdf5_path=$DATASPACE"/hdf5s/waveforms/eval_19393_files_kaiser_best.h5"

# # Pack balanced training waveforms to a single hdf5 file
# python3 $SCRIPT pack_waveforms_to_hdf5 \
#     --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" \
#     --audios_dir=$DATASET_DIR"/audios/balanced_train_segments" \
#     --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5"

# # Pack unbalanced training waveforms to hdf5 files. Users may consider 
# # executing the following commands in parallel to speed up. One simple 
# # way is to open 41 terminals and execute one command in one terminal.
# for IDX in {00..40}; do
#     echo $IDX
#     python3 $SCRIPT pack_waveforms_to_hdf5 \
#         --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" \
#         --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX" \
#         --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5"
# done

# Pack AudioCaps waveforms to a single hdf5 file

AUDIO_DIR=/gpfsscratch/rech/xsl/commun/data/AUDIOCAPS_32000Hz
BASEDIR=/gpfsstore/rech/djl/uzj43um/audiocaps

subset=val

srun $PYTHON -u $SCRIPT pack_waveforms_to_hdf5 \
    --csv_path=$BASEDIR"/${subset}.csv" \
    --audios_dir=$AUDIO_DIR"/audio/${subset}" \
    --waveforms_hdf5_path=$BASEDIR"/hdf5s/waveforms/${subset}.h5"

