#!/bin/bash

DNAME_RUN=$1
FNAME_PYSCRIPT=$2
FNAME_SBATCH=$3

# Create run dir if not existing yet
# rundir=runs_pretraining_from_scratch
rundir=../runs

mkdir -p $rundir/$DNAME_RUN/ || exit 1

# Save .sbatch and .py into run dir
cp $FNAME_SBATCH ../pytorch/$FNAME_PYSCRIPT ../pytorch/models.py $rundir/$DNAME_RUN/

# Move to run dir
cd $rundir/$DNAME_RUN/

# Create log dir
# mkdir sbatch_logs/

# Start job
# sbatch $FNAME_SBATCH $FNAME_PYSCRIPT
sbatch $FNAME_SBATCH 

# examples
# $ bash ./launch_4.sh cnn14_fulltrain_balanced_clipbce_seed1978_bs64_lr001_max100000 main.py 4_train.sbatch
# $ bash ./launch_4.sh cnn14deformable_fulltrain_balanced_clipbce_seed1978_bs64_lr001_max100000_block1 main.py 4_train.sbatch


