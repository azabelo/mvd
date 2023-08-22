#!/bin/bash

# Check if four arguments are provided
if [ $# -ne 8 ]; then
    echo "please provide rid:GPS rid:MASTER_PORT rid:MASTER_ADDR (localhost when using only 1 GPU) BATCH_SIZE INITIAL_LR UPDATE_FREQ EPOCHS WARMUP SAMPLING_RATE USE_CLIP CHECKPOINT"
    exit 1
fi
#NODE_COUNT RANK
GPUS=1
#NODE_COUNT="$2"
#RANK="$3"
MASTER_PORT=1
#MASTER_ADDR="$3"
BATCH_SIZE="$1"
LEARNING_RATE="$2"
UPDATE_FREQ="$3"
EPOCHS="$4"
WARMUP="$5"
SAMPLING_RATE="$6"
USE_CLIP="$7"
CHECKPOINT="$8"
OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_HMDB51'
DATA_PATH='train.csv'
DATA_ROOT='hmdb51_mp4'

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS} \
        --master_port ${MASTER_PORT} --nnodes=1 \
        --node_rank=0 --master_addr=localhost \
        run_mvd_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_root ${DATA_ROOT} \
        --model pretrain_masked_video_student_base_patch16_224 \
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
        --batch_size ${BATCH_SIZE} --update_freq ${UPDATE_FREQ} --save_ckpt_freq 20 \
        --num_frames 16 --sampling_rate ${SAMPLING_RATE} \
        --lr ${LEARNING_RATE} --min_lr 1e-4 --drop_path 0.1 --warmup_epochs ${WARMUP} --epochs ${EPOCHS} \
        --auto_resume \
        --use_cls_token \
        --use_clip ${USE_CLIP} --use_checkpoint --load_model ${CHECKPOINT} --checkpoint_path ${CHECKPOINT}