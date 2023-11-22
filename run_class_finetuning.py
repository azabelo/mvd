import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict


from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from timm.data import Mixup
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets import build_dataset
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate
import utils
import modeling_finetune
import torch.nn as nn

import wandb
from s3d import S3D

import clip
from timm.models import create_model
import modeling_student
import modeling_teacher
import modeling_video_teacher
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser('MVD fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    parser.add_argument('--remove_pos_emb', action='store_true', default=False)

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    parser.add_argument('--use_cls_token', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_root', default=None, type=str,
                        help='dataset path root')
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='path of dataset file list')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51','image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume_best', action='store_true', default=False)
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--no_save_best_ckpt', action='store_true', default=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--merge_test', action='store_true', default=False)
    parser.add_argument('--eval_log_name', default='log_eval', type=str,
                        help='Perform evaluation only, the log name')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--use_clip', default=0, type=int)

    parser.add_argument('--additional_name', default='', type=str)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    print(args.start_epoch)
    run_name = f"bs: {args.batch_size}, update: {args.update_freq}, lr: {args.lr}, epochs: {args.epochs}, \
    warmup: {args.warmup_epochs}, sampling: {args.sampling_rate}, segments: {args.test_num_segment}, \
    crops: {args.test_num_crop} {args.additional_name}"
    if args.use_clip:
        run_name = "CLIP " + run_name
    else:
        run_name = "MAE " + run_name

        # 'testing zero shot'
    wandb.init(project='MVD finetuning test', name=run_name)
    # Log the arguments to wandb
    wandb.config.update(args)

    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # # only for linear probing
    # import torch.nn as nn
    # class ExtendedModel(nn.Module):
    #     def __init__(self, base_model, linear_size):
    #         super(ExtendedModel, self).__init()
    #
    #         # Load the pre-trained model and freeze it
    #         self.base_model = base_model
    #         for param in self.base_model.parameters():
    #             param.requires_grad = False
    #
    #         # Add a trainable linear layer
    #         self.linear_layer = nn.Linear(32 * 51, 32 * 51)
    #
    #     def forward(self, x):
    #         # Pass the input through the base model
    #         features = self.base_model(x)

    class S3DClassifier(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()

            # Load the S3D video encoder
            self.video_encoder = S3D('pretrained_models/s3d_dict.npy', 512)
            self.video_encoder.load_state_dict(torch.load('pretrained_models/s3d_howto100m.pth'))

            # Add a linear layer for classification
            self.classification_layer = torch.nn.Linear(512, num_classes)

        def forward(self, video_frames):
            # Forward pass through the video encoder
            video_embedding = self.video_encoder(video_frames)['video_embedding']

            # Apply the classification linear layer
            output = self.classification_layer(video_embedding)

            return output

    model = None
    if args.finetune == 's3d':
        model = S3DClassifier(args.nb_classes)

    else:
        model = create_model(
            args.model,
            pretrained=False,
            img_size=args.input_size,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_cls_token=args.use_cls_token,
            fc_drop_rate=args.fc_drop_rate,
            use_checkpoint=args.use_checkpoint,
        )




        patch_size = model.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size

    # only if linear probing!!!!!!!!!!!!!!!!!!!!!!!!#

    # for name, param in model.named_parameters():
    #     if name == "head.weight" or name == "head.bias":
    #         param.requires_grad = True
    #         print(f"Layer Name: {name}, Shape: {param.shape}")
    #     else:
    #         param.requires_grad = False


    # linear probing with an additional linear layer:

    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    # intermediate_layer = nn.Linear(768, 768)  # Map 768 to 768
    # model.head.weight.requires_grad = False
    # model.head.bias.requires_grad = False
    # new_head = nn.Linear(768, 51)  # 51 is the number of output classes
    # # Replace the existing head with the new head
    # model.head = nn.Sequential(intermediate_layer, new_head)




    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None





    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.finetune != 's3d':
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if args.remove_pos_emb and 'pos_embed' in key:
                continue
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding 
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    # # print(model)
    # # print(type(model))
    #
    # def print_layer_weights(model, layer_name):
    #     layer = getattr(model, layer_name, None)
    #     if layer is not None:
    #         print(f"Weights of {layer_name}:")
    #         print(layer.weight)
    #     else:
    #         print(f"Layer {layer_name} not found in the model.")
    # # print(model.state_dict()["blocks.11.mlp.fc2.weight"])
    # # print(model.state_dict().keys())
    # for key in model.state_dict().keys():
    #     temp_tensor = model.state_dict()[key]
    #     while isinstance(temp_tensor, torch.Tensor) and temp_tensor.dim() > 0:
    #         temp_tensor =  temp_tensor[0]
    #     print(key, model.state_dict()[key].shape, temp_tensor)


    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)



    if args.finetune == 's3d':
        skip_weight_decay_list = ()
        assigner = None

        if args.enable_deepspeed:
            loss_scaler = None
            optimizer_params = get_parameter_groups(
                model, args.weight_decay, skip_weight_decay_list,
                assigner.get_layer_id if assigner is not None else None,
                assigner.get_scale if assigner is not None else None)
            model, optimizer, _, _ = ds_init(
                args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
            )

            print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
            assert model.gradient_accumulation_steps() == args.update_freq
        else:
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
                model_without_ddp = model.module

            optimizer = create_optimizer(
                args, model_without_ddp, skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None,
                get_layer_scale=assigner.get_scale if assigner is not None else None)
            loss_scaler = NativeScaler()

    else:
        num_layers = model_without_ddp.get_num_layers()

        if args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()
        print("Skip weight decay list: ", skip_weight_decay_list)

        if args.enable_deepspeed:
            loss_scaler = None
            optimizer_params = get_parameter_groups(
                model, args.weight_decay, skip_weight_decay_list,
                assigner.get_layer_id if assigner is not None else None,
                assigner.get_scale if assigner is not None else None)
            model, optimizer, _, _ = ds_init(
                args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
            )

            print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
            assert model.gradient_accumulation_steps() == args.update_freq
        else:
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                model_without_ddp = model.module

            optimizer = create_optimizer(
                args, model_without_ddp, skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None,
                get_layer_scale=assigner.get_scale if assigner is not None else None)
            loss_scaler = NativeScaler()



    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.output_dir and utils.is_main_process():
        config_name = args.eval_log_name + '_config.txt' if args.eval else "config.txt"
        with open(os.path.join(args.output_dir, config_name), mode="a", encoding="utf-8") as f:
            for arg in vars(args):
                f.write(format(arg, '<20') + " " + format(str(getattr(args, arg)), '<') + "\n")  # str, arg_type

    if args.eval:
        if not args.merge_test:
            preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
            test_stats = final_test(data_loader_test, model, device, preds_file)
            torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, args.eval_log_name + ".txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    ######## inital knn for failsafe blyat ##########

    import engine_for_pretraining
    import copy

    args2 = copy.deepcopy(args)
    args2.data_set = 'HMDB51'
    args2.nb_classes = 51
    args2.data_path = 'finetune_splits'
    args2.test_num_segment = 8
    args2.test_num_crop = 1
    args2.short_side_size = 256
    args2.batch_size = 8
    dataset_val2, _ = build_dataset(is_train=False, test_mode=False, args=args2)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # dont forget that you added shuffle and took something out
    data_loader_val2 = torch.utils.data.DataLoader(
        dataset_val2,
        batch_size=int(args2.batch_size),
        num_workers=args2.num_workers,
        pin_memory=args2.pin_mem,
        drop_last=False,
        shuffle=True
    )
    print(type(model))
    engine_for_pretraining.log_knn_acc(data_loader_val2, model, finetuning=True)



    ######### for zero shot ############


    zero_shot_blyat = False

    clip_model = None
    vision_encoder = None
    text_features = None

    if zero_shot_blyat:
        clip_model, preprocess = clip.load("ViT-B/16", device=device)
        #vision_encoder = get_4799()
        vision_encoder = get4799student()
        text_features, prompts = get_text_embs()




    print("start epoch: ", args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        # if data_loader_val is not None:
        #     test_stats = validation_one_epoch(data_loader_val, model, device)
        #     print(f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%")
        #     if max_accuracy < test_stats["acc1"]:
        #         max_accuracy = test_stats["acc1"]
        #         if args.output_dir and args.save_ckpt and not args.no_save_best_ckpt:
        #             utils.save_model(
        #                 args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #                 loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
        #
        #     print(f'Max accuracy: {max_accuracy:.2f}%')


        if zero_shot_blyat:
            train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
                zero_shot_blyat=zero_shot_blyat,
                clip_model=clip_model, vision_encoder=vision_encoder, text_features=text_features, prompts=prompts,
            )
            continue


        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, zero_shot_blyat=zero_shot_blyat,
            clip_model=clip_model, vision_encoder=vision_encoder, text_features=text_features, args=args
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = validation_one_epoch(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt and not args.no_save_best_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(val_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(val_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(data_loader_test, model, device, preds_file)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1,
                    'Final Top-5': final_top5}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_text_embs():
    templates = [
        'a photo of a person {}.',
        'a video of a person {}.',
        'a example of a person {}.',
        'a demonstration of a person {}.',
        'a photo of the person {}.',
        'a video of the person {}.',
        'a example of the person {}.',
        'a demonstration of the person {}.',
        'a photo of a person using {}.',
        'a video of a person using {}.',
        'a example of a person using {}.',
        'a demonstration of a person using {}.',
        'a photo of the person using {}.',
        'a video of the person using {}.',
        'a example of the person using {}.',
        'a demonstration of the person using {}.',
        'a photo of a person doing {}.',
        'a video of a person doing {}.',
        'a example of a person doing {}.',
        'a demonstration of a person doing {}.',
        'a photo of the person doing {}.',
        'a video of the person doing {}.',
        'a example of the person doing {}.',
        'a demonstration of the person doing {}.',
        'a photo of a person during {}.',
        'a video of a person during {}.',
        'a example of a person during {}.',
        'a demonstration of a person during {}.',
        'a photo of the person during {}.',
        'a video of the person during {}.',
        'a example of the person during {}.',
        'a demonstration of the person during {}.',
        'a photo of a person performing {}.',
        'a video of a person performing {}.',
        'a example of a person performing {}.',
        'a demonstration of a person performing {}.',
        'a photo of the person performing {}.',
        'a video of the person performing {}.',
        'a example of the person performing {}.',
        'a demonstration of the person performing {}.',
        'a photo of a person practicing {}.',
        'a video of a person practicing {}.',
        'a example of a person practicing {}.',
        'a demonstration of a person practicing {}.',
        'a photo of the person practicing {}.',
        'a video of the person practicing {}.',
        'a example of the person practicing {}.',
        'a demonstration of the person practicing {}.',
    ]


    templates = [
        'a photo of a person {}.',
]

    class_names_str = "brush_hair clap draw_sword fall_floor handstand kick pick push run shoot_gun smoke sword turn cartwheel climb dribble fencing hit kick_ball pour pushup shake_hands sit somersault sword_exercise walk catch climb_stairs drink flic_flac hug kiss pullup ride_bike shoot_ball situp stand talk wave chew dive eat golf jump laugh punch ride_horse shoot_bow smile swing_baseball throw"
    action_str = "brushing_hair claping drawing_a_sword falling_to_the_floor doing_a_handstand kicking picking pushing running shooting_a_gun smoking using_a_sword turning doing_a_cartwheel climbing dribbling fencing hitting_something kicking_a_ball pouring doing_pushups shaking_hands sitting doing_a_somersault doing_sword_exercises walking catching climbing_stairs drinking flic_flacing hugging kissing doing_pullups riding_a_bike shooting_a_ball doing_situps standing talking waving chewing diving eating golfing jumping laughing punching riding_a_horse shooting_a_bow smiling swinging_a_baseball_bat throwing"

    # Convert the string to a list by splitting on spaces and then removing underscores
    class_names = action_str.split()
    # Sort the list alphabetically
    class_names.sort()

    prompts = []
    for name in class_names:
        name = name.replace('_', ' ')
        for template in templates:
            prompts.append(template.format(name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    text_embs = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(text_embs)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features, prompts

def get_4799():

    video_teacher_model_ckpt_path = 'checkpoint-4799.pth'
    model_key = "model|module"

    video_teacher_model = create_model('pretrain_videomae_teacher_base_patch16_224', pretrained=False, img_size=224,
                                       drop_path_rate=0)
    checkpoint = torch.load(video_teacher_model_ckpt_path, map_location='cpu')

    if video_teacher_model_ckpt_path:

        checkpoint = torch.load(video_teacher_model_ckpt_path, map_location='cpu')

        print("Load video teacher ckpt from %s" % video_teacher_model_ckpt_path)
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load video state_dict by model_key = %s" % model_key)
                break

        if checkpoint_model is None:
            checkpoint_model = checkpoint

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        if video_teacher_model_ckpt_path == 'video_teacher.pth':
            for key in all_keys:
                print("video_teacher: ", key)
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif 'pos_embed' in key:
                    continue
                else:
                    new_dict[key] = checkpoint_model[key]
        elif video_teacher_model_ckpt_path == 'checkpoint-4799.pth':
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif 'pos_embed' in key:
                    continue
                else:
                    new_dict["encoder." + key] = checkpoint_model[key]
        elif video_teacher_model_ckpt_path == 'vit_b_k710_dl_from_giant.pth':
            for key in all_keys:
                print("v2 teacher: ", key)
                if key == 'fc_norm.weight':
                    new_dict["encoder.norm.weight"] = checkpoint_model[key]
                    continue
                elif key == 'fc_norm.bias':
                    new_dict["encoder.norm.bias"] = checkpoint_model[key]
                    continue

                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif 'pos_embed' in key:
                    continue
                else:
                    new_dict["encoder." + key] = checkpoint_model[key]

        checkpoint_model = new_dict

        utils.load_state_dict(video_teacher_model, checkpoint_model, prefix='')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_teacher_model.to(device)
    return video_teacher_model

def get4799student():
    print(f"Creating model: 4799")
    model = create_model(
        'pretrain_masked_video_student_base_patch16_224',
        pretrained=False,
        drop_path_rate=0.1,
        drop_block_rate=None,
        decoder_depth=2,
        use_cls_token=True,
        num_frames=16,
        target_feature_dim=768,
        target_video_feature_dim=768,
        feat_decoder_embed_dim=None,
        feat_decoder_num_heads=None,
        # use_checkpoint=args.use_checkpoint,
        # checkpoint_path=args.checkpoint_path,
        tubelet_size=2,
    )

    weights = torch.load('checkpoint-4799.pth', map_location='cpu')['model']
    utils.load_state_dict(model, weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model

if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
