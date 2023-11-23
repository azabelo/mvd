#!/bin/bash

# Check if four arguments are provided
if [ $# -ne 9 ]; then
    echo "please provide GPS MASTER_PORT rid:MASTER_ADDR (localhost when using only 1 GPU) BATCH_SIZE INITIAL_LR UPDATE_FREQ EPOCHS WARMUP SAMPLING_RATE USE_CLIP"
    exit 1
fi
#NODE_COUNT RANK
GPUS="$1"
#NODE_COUNT="$2"
#RANK="$3"
MASTER_PORT="$2"
#MASTER_ADDR="$3"
BATCH_SIZE="$3"
LEARNING_RATE="$4"
UPDATE_FREQ="$5"
EPOCHS="$6"
WARMUP="$7"
SAMPLING_RATE="$8"
USE_CLIP="$9"


OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_base_teacher_HMDB51'
DATA_PATH='clean_train.csv'
DATA_ROOT='hmdb51_mp4'
#DATA_PATH='tiny_k400_train.csv'
#DATA_ROOT='tiny-Kinetics-400'

#checkpoint-4799.pth
#video_teacher.pth
#vit_b_k710_dl_from_giant.pth

#     --use_checkpoint     --checkpoint_path OUTPUT/mvd_vit_base_with_vit_base_teacher_HMDB51/checkpoint-399.pth


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
        --video_teacher_model_ckpt_path 'vit_b_k710_dl_from_giant.pth' \
        --mask_type tube --mask_ratio 0.9 --decoder_depth 2 \
        --batch_size ${BATCH_SIZE} --update_freq ${UPDATE_FREQ} --save_ckpt_freq 200 \
        --num_frames 16 --sampling_rate ${SAMPLING_RATE} \
        --lr ${LEARNING_RATE} --min_lr 1e-4 --drop_path 0.1 --warmup_epochs ${WARMUP} --epochs ${EPOCHS} \
        --use_clip ${USE_CLIP} --use_cls_token --auto_resume --norm_feature \

