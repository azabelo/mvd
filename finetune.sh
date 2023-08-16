#!/bin/bash

# Check if four arguments are provided
if [ $# -ne 9 ]; then
    echo "please provide GPS MASTER_PORT MASTER_ADDR (localhost when using only 1 GPU) BATCH_SIZE LR MODEL_PATH EPCHOS UPDATE_FREQ NUM_SAMPLES"
    exit 1
fi

GPUS="$1"
MASTER_PORT="$2"
MASTER_ADDR="$3"
BATCH_SIZE="$4"
LEARNING_RATE="$5"
OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_HMDB51_finetune'
MODEL_PATH="$6"
DATA_PATH='finetune_splits'
DATA_ROOT='hmdb51_mp4'
EPOCHS="$7"
UPDATE_FREQ="$8"
NUM_SAMPLES="$9"


OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} --nnodes=1 \
    --node_rank=0 --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 --nb_classes 51 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size ${BATCH_SIZE} --update_freq ${UPDATE_FREQ} --num_sample ${NUM_SAMPLES} \
    --save_ckpt_freq 20 --no_save_best_ckpt \
    --num_frames 16 \
    --lr ${LEARNING_RATE} --epochs ${EPOCHS} \
    --dist_eval --test_num_segment 2 --test_num_crop 3 \
    --use_checkpoint \
    --enable_deepspeed
