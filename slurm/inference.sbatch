#!/usr/bin/env bash

#SBATCH --output=jobs/gst/inference_%J_slurm.out
#SBATCH --error=jobs/gst/inference_%J_slurm.out
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

CKPT_PATH=/Midgard/home/martinig/thesis-src/jobs/gst/364600/lightning_logs/version_364600/checkpoints/epoch=0-step=13852.ckpt

export TRANSFORMERS_OFFLINE=1 # does NOT affect pytorch_bert_pretrained
python code/inference.py --checkpoint_path $CKPT_PATH