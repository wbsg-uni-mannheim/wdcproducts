#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
MODEL=$1
CHECKPOINT=$2
BATCH=$3
LR=$4
CATEGORY=$5
SIZE=$6
AUG=$7
python run_finetune_baseline_multi.py \
    --do_train \
    --train_file ../../data/interim/wdc-lspc/training-sets/preprocessed_${CATEGORY}_train_$SIZE.pkl.gz \
	--train_size=$SIZE \
	--validation_file ../../data/interim/wdc-lspc/validation-sets/preprocessed_${CATEGORY}_valid_$SIZE.pkl.gz \
	--test_file ../../data/interim/wdc-lspc/gold-standards/preprocessed_${CATEGORY}_gs.pkl.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
    --output_dir ../../reports/baseline-multi/$CATEGORY-$SIZE-$AUG$BATCH-$LR-${MODEL##*/}/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1_micro \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	#--do_param_opt \