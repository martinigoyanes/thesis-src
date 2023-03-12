#!/usr/bin/env bash

#SBATCH --output=jobs/gst/evaluation_%J_slurm.out
#SBATCH --error=jobs/gst/evaluation_%J_slurm.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="arwen"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH -t 0-6:00  # time limit: (D-HH:MM) 


# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate thesis-src


JOB_PATH=/Midgard/home/martinig/thesis-src/jobs/gst/364600
# use references as input
INPUT_PATH=/Midgard/home/martinig/thesis-src/data/yelp/bert_best_head_removal/reference.0.out
# use orignial as input
# INPUT_PATH=/Midgard/home/martinig/thesis-src/data/yelp/test_toxic.txt

BATCH_SIZE=32
TASK_NAME=yelp

export TRANSFORMERS_OFFLINE=1
python evaluation/metric.py  --inputs $INPUT_PATH --preds "$JOB_PATH/preds.txt" --batch_size 32 --out_dir $JOB_PATH --task_name $TASK_NAME