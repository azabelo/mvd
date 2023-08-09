#!/bin/bash

# Check if four arguments are provided
if [ $# -ne 7 ]; then
    echo "please provide GPS NODE_COUNT RANK MASTER_PORT MASTER_ADDR (localhost when using only 1 GPU) BATCH_SIZE INITIAL_LR"
    exit 1
fi

GPUS="$1"
NODE_COUNT="$2"
RANK="$3"
MASTER_PORT="$4"
MASTER_ADDR="$5"
BATCH_SIZE="$6"
LEARNING_RATE="$7"
OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_HMDB51'
DATA_PATH='train.csv'
DATA_ROOT='hmdb51_mp4'

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS} \
        --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
        --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
        run_mvd_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_root ${DATA_ROOT} \
        --model pretrain_masked_video_student_base_patch16_224 \
        --opt adamw --opt_betas 0.9 0.95 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --image_teacher_model vit_base_patch16_224 \
        --distillation_target_dim 768 \
        --distill_loss_func SmoothL1 \
        --image_teacher_model_ckpt_path 'image_teacher.pth' \
        --video_teacher_model pretrain_videomae_teacher_base_patch16_224 \
        --video_distillation_target_dim 768 \
        --video_distill_loss_func SmoothL1 \
        --video_teacher_model_ckpt_path 'video_teacher.pth' \
        --mask_type tube --mask_ratio 0.9 --decoder_depth 2 \
        --batch_size ${BATCH_SIZE} --update_freq 2 --save_ckpt_freq 3 \
        --num_frames 16 --sampling_rate 4 \
        --lr ${LEARNING_RATE} --min_lr 1e-2 --drop_path 0.1 --warmup_epochs 3 --epochs 10 \
        --auto_resume