#!/bin/bash

# Check if four arguments are provided
if [ $# -ne 8 ]; then
    echo "please provide GPS NODE_COUNT RANK MASTER_PORT MASTER_ADDR (localhost when using only 1 GPU) BATCH_SIZE INITIAL_LR MODEL_PATH"
    exit 1
fi

GPUS="$1"
NODE_COUNT="$2"
RANK="$3"
MASTER_PORT="$4"
MASTER_ADDR="$5"
BATCH_SIZE="$6"
LEARNING_RATE="$7"
OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_HMDB51_finetune'
MODEL_PATH="$8"
DATA_PATH='train.csv'
DATA_ROOT='hmdb51_mp4'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
    --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set HMDB51 --nb_classes 51 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size ${BATCH_SIZE} --update_freq 1 --num_sample 2 \
    --save_ckpt_freq 5 --no_save_best_ckpt \
    --num_frames 16 \
    --lr 5e-4 --epochs 10 \
    --dist_eval --test_num_segment 2 --test_num_crop 3 \
    --use_checkpoint \
    --enable_deepspeed