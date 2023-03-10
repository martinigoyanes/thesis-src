#!/usr/bin/env bash

#SBATCH --output=jobs/bart-detox/evaluation/%J_slurm.out
#SBATCH --error=jobs/bart-detox/evaluation/%J_slurm.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="rivendell"
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


JOB_PATH=/Midgard/home/martinig/thesis-src/jobs/gst/training/${SLURM_JOB_ID}
mkdir -p $JOB_PATH

BATCH_SIZE=32
PREPROCESS_KIND=bert_best_head_removal
MODEL_NAME=BlindGST
DATAMODULE_NAME=YelpDM

export TRANSFORMERS_OFFLINE=1
python code/train_gst.py --batch_size $BATCH_SIZE --datamodule_name $DATAMODULE_NAME --model_name $MODEL_NAME --default_root_dir $JOB_PATH --preprocess_kind $PREPROCESS_KIND