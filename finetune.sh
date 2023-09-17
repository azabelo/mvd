#!/bin/bash

# Check if four arguments are provided
if [ $# -ne 10 ]; then
    echo "please provide GPS MASTER_PORT rid:MASTER_ADDR (localhost when using only 1 GPU) BATCH_SIZE LR MODEL_PATH EPOCHS UPDATE_FREQ NUM_SAMPLES CLIP_USED WARMUP"
    exit 1
fi

GPUS="$1"
MASTER_PORT="$2"
#MASTER_ADDR="$3"
BATCH_SIZE="$3"
LEARNING_RATE="$4"
OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_HMDB51_finetune'
MODEL_PATH="$5"
DATA_PATH='official_splits'
DATA_ROOT='hmdb51_mp4'
EPOCHS="$6"
UPDATE_FREQ="$7"
NUM_SAMPLES="$8"
USE_CLIP="$9"
WARMUP="${10}"
#     --finetune ${MODEL_PATH} \     --use_checkpoint \
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} --nnodes=1 \
    --node_rank=0 --master_addr=localhost \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 --nb_classes 51 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size ${BATCH_SIZE} --update_freq ${UPDATE_FREQ} --num_sample ${NUM_SAMPLES} \
    --save_ckpt_freq 50 \
    --num_frames 16 \
    --lr ${LEARNING_RATE} --epochs ${EPOCHS} \
    --dist_eval --test_num_segment 10 --test_num_crop 3 \
    --enable_deepspeed --warmup_epochs ${WARMUP} \
    --use_clip ${USE_CLIP} \
    --layer_decay 0.7 --drop_path 0.2 --no_auto_resume

