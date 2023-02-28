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


JOB_PATH=/Midgard/home/martinig/thesis-src/jobs/bart-detox/evaluation/${SLURM_JOB_ID}
PREDS_PATH=
INPUT_PATH=

export TRANSFORMERS_OFFLINE=1
python emnlp2021/style_transfer/paraGeDi/gedi_training/gedi_training.py \
--tokenizer_name $TOKENIZER_NAME \
--data_dir $DATA_PATH \
--working_dir "${JOB_PATH}/model" \
--log_dir "${JOB_PATH}/logs" \
--n_epochs $N_EPOCHS \
--max_seq_length $MAX_SEQUENCE_LEN \
--train_batch_size $BATCH_SIZE \
--eval_batch_size $BATCH_SIZE \
--logging_steps $LOGGING_STEPS \
--saving_steps $SAVING_STEPS \
--tensorboard