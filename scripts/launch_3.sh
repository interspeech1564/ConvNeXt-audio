#!/bin/bash

# Start job
# sbatch $FNAME_SBATCH $FNAME_PYSCRIPT

#for IDX in {01..10}; do
for IDX in { 01 02 03 04 05 06 07 08 09 10 }; do
 echo $IDX
 sbatch 3_create_training_indexes_template.sbatch $IDX
done

# examples
# $ bash ./launch.sh preproc_val run_wav2vec2_feature_extraction.py launch_preproc.sbatch
# $ bash ./launch.sh pretrain_from_scratch_2GPUs run_wav2vec2_pretraining_FSD50K_from_scratch.py launch_pretraining_nGPUs.sbatch
