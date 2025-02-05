import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import torchvision
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import wandb
import torchvision.transforms.functional as TF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor
import torch.nn.functional as F

Loss_func_choice = {'L1': torch.nn.L1Loss, 'L2': torch.nn.MSELoss, 'SmoothL1': torch.nn.SmoothL1Loss}

def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, update_freq=None, time_stride_loss=True, lr_scale=1.0,
                    image_teacher_model=None, video_teacher_model=None, norm_feature=False,data_for_knn=None,):

    # test that the output of the video teacher doesn't change by passing in a ones vector
    ones_video_features = video_teacher_model(torch.ones((1, 3, 16, 224, 224)).cuda())
    print("ones video features (epoch start): ", ones_video_features[:,0,:25])

    if data_for_knn is not None:
        log_knn_acc(data_for_knn, model)
        # test that the output of the video teacher doesn't change by passing in a ones vector
        ones_video_features = video_teacher_model(torch.ones((1, 3, 16, 224, 224)).cuda())
        print("ones video features (after knn): ", ones_video_features[:, 0, :25])

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    LN_img = nn.LayerNorm(args.distillation_target_dim, eps=1e-6, elementwise_affine=False).cuda()
    LN_vid = nn.LayerNorm(args.video_distillation_target_dim, eps=1e-6, elementwise_affine=False).cuda()

    loss_func_img_feat = Loss_func_choice[args.distill_loss_func]()
    loss_func_vid_feat = Loss_func_choice[args.video_distill_loss_func]()
    image_loss_weight = args.image_teacher_loss_weight
    video_loss_weight = args.video_teacher_loss_weight

    tubelet_size = args.tubelet_size



    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        update_step = step // update_freq
        it = start_steps + update_step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None and step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    #print("it: ", it)
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"] * lr_scale
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, videos_for_teacher, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        videos_for_teacher = videos_for_teacher.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        B, C, T, H, W = videos.shape
        print("video shape , ", videos.shape)
        print("tubelet size ", tubelet_size)

        assert H == W == 224, "input size must be 224 for CLIP"

        #normalization used in CLIP
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        mean_tensor = torch.tensor(clip_mean)
        std_tensor = torch.tensor(clip_std)

        with torch.cuda.amp.autocast():
            output_features, output_video_features = model(videos, bool_masked_pos)
            with torch.no_grad():
                image_teacher_model.eval()
                print("model name: ", image_teacher_model.__class__.__name__)
                if time_stride_loss:
                    teacher_features = image_teacher_model(
                        TF.normalize(rearrange(videos_for_teacher[:, :, ::tubelet_size, :, :], 'b c t h w -> (b t) c h w'),
                                    mean=clip_mean, std=clip_std)
                    )
                    print("raw image feats: ", teacher_features.shape)
                    teacher_features = rearrange(teacher_features, '(b t) l c -> b (t l) c', t=T//tubelet_size)
                else:
                    teacher_features = image_teacher_model(
                        rearrange(videos_for_teacher, 'b c t h w -> (b t) c h w'),
                    )
                    print("raw image feats: ", teacher_features.shape)
                    teacher_features = rearrange(teacher_features, '(b t d) l c -> b (t l) (d c)', t=T//tubelet_size, d=tubelet_size)
                if norm_feature:
                    teacher_features = LN_img(teacher_features)

                video_teacher_model.eval()
                videos_for_video_teacher = videos if args.video_teacher_input_size == args.input_size \
                    else videos_for_teacher

                video_teacher_features = video_teacher_model(videos_for_video_teacher)
                if norm_feature:
                    video_teacher_features = LN_vid(video_teacher_features)

                print("video teacher features shape: ", video_teacher_features.shape)
                print("image teacher features shape: ", teacher_features.shape)

            B, _, D = output_features.shape
            loss_img_feat = loss_func_img_feat(
                input=output_features,
                target=teacher_features[bool_masked_pos].reshape(B, -1, D)
            )
            loss_value_img_feat = loss_img_feat.item()

            print("video feature shape: ", output_video_features.shape)
            print("image feature shape: ", output_features.shape)
            B, _, D = output_video_features.shape
            loss_vid_feat = loss_func_vid_feat(
                input=output_video_features,
                target=video_teacher_features[bool_masked_pos].reshape(B, -1, D)
            )
            loss_value_vid_feat = loss_vid_feat.item()

            loss = image_loss_weight * loss_img_feat + video_loss_weight * loss_vid_feat

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(step + 1) % update_freq == 0)
        if (step + 1) % update_freq == 0:
            optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_img_feat=loss_value_img_feat)
        metric_logger.update(loss_vid_feat=loss_value_vid_feat)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        wandb.log({"epoch": epoch, "batch": step, "train_loss": loss_value, " train_img_feat_loss": loss_value_img_feat,
           "min_lr": min_lr, "max_lr": max_lr, "train_vid_feat_loss": loss_value_vid_feat, "grad_norm": grad_norm,})

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_img_feat=loss_value_img_feat, head="loss_img_feat")
            log_writer.update(loss_vid_feat=loss_value_vid_feat, head="loss_vid_feat")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)



    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def log_knn_acc(data_for_knn, model, finetuning=False):

    # lightly knn
    train_videos = torch.empty((0, 768), device='cuda')
    test_videos = torch.empty((0, 768), device='cuda')
    train_labels = torch.empty(0, device='cuda')
    test_labels = torch.empty(0, device='cuda')

    # my implementation of knn
    knn_classifier3 = KNeighborsClassifier(n_neighbors=3, algorithm='brute', metric='cosine')
    train_videos_np = np.empty((0, 768))
    test_videos_np = np.empty((0, 768))
    train_labels_np = np.empty(0)
    test_labels_np = np.empty(0)

    with torch.no_grad():
        index = 0
        for batch in data_for_knn:
            print("knn step: ", index)
            index += 1
            videos, labels, _ = batch
            # make an empty tensor of False values with shape [bs, 1568]
            empty_mask = torch.zeros((videos.shape[0], 1568), dtype=torch.bool)
            cls_tok_knn = None
            if finetuning:
                # forward features already takes the cls token
                # output_features_video_for_knn = model.forward_features(videos.cuda())
                output_features_video_for_knn = model(videos.unsqueeze(0).cuda())
                cls_tok_knn = output_features_video_for_knn.cuda()
                print(output_features_video_for_knn.shape)
            else:
                output_features_for_knn, output_features_video_for_knn = model(videos.cuda(), empty_mask.cuda())
            # output_features_video_for_knn = output_features_video_for_knn.cpu().numpy()
                cls_tok_knn = output_features_video_for_knn[:, 0, :]
                cls_tok_knn = F.normalize(cls_tok_knn, dim=1)
                cls_tok_knn = cls_tok_knn.cuda()
            if index > 100:
                # move to cuda if not already
                test_labels = test_labels.cuda()
                test_videos = test_videos.cuda()
                labels = labels.cuda()
                cls_tok_knn = cls_tok_knn.cuda()
                test_labels = torch.cat((test_labels, labels), 0)
                test_videos = torch.cat((test_videos, cls_tok_knn), 0)
                # test_videos_np = np.append(test_videos, output_features_video_for_knn.reshape(8, -1), axis=0)
                test_labels_np = np.append(test_labels_np, labels.cpu().numpy(), axis=0)
                test_videos_np = np.append(test_videos_np, cls_tok_knn.cpu().numpy(), axis=0)
            else:
                train_labels = train_labels.cuda()
                train_videos = train_videos.cuda()
                labels = labels.cuda()
                cls_tok_knn = cls_tok_knn.cuda()
                train_labels = torch.cat((train_labels, labels), 0)
                train_videos = torch.cat((train_videos, cls_tok_knn), 0)
                # train_videos_np = np.append(train_videos, output_features_video_for_knn.reshape(8, -1), axis=0)
                train_labels_np = np.append(train_labels_np, labels.cpu().numpy(), axis=0)
                train_videos_np = np.append(train_videos_np, cls_tok_knn.cpu().numpy(), axis=0)

        # custom knn
        # Standardize the feature values
        scaler = StandardScaler()
        train_scaled_np = scaler.fit_transform(train_videos_np)
        test_scaled_np = scaler.transform(test_videos_np)
        knn_classifier3.fit(train_scaled_np, train_labels_np)
        predictions3 = knn_classifier3.predict(test_scaled_np)
        knn_accuracy_custom = accuracy_score(test_labels_np, predictions3)
        print("custom knn accuracy", knn_accuracy_custom)

        # lightly knn
        #switch dimensions of the train_videos
        train_videos = train_videos.transpose(0, 1)
        pred_labels = knn_predict(
            test_videos,
            train_videos,
            train_labels,
            num_classes=51,
        )
        print(pred_labels.shape)
        print(test_labels.shape)
        print(pred_labels[0])
        test_labels = test_labels.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        lightly_knn_accuracy = accuracy_score(test_labels, pred_labels)

        wandb.log({"knn_accuracy_lightly": lightly_knn_accuracy, "knn_accuracy_custom": knn_accuracy_custom})

def knn_predict(
    feature: Tensor,
    feature_bank: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    knn_k: int = 199,
    knn_t: float = 0.1,
) -> Tensor:
    """Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature:
            Tensor with shape (B, D) for which you want predictions.
        feature_bank:
            Tensor of shape (D, N) of a database of features used for kNN.
        feature_labels:
            Labels with shape (N,) for the features in the feature_bank.
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10).
        knn_k:
            Number of k neighbors used for kNN.
        knn_t:
            Temperature parameter to reweights similarities for kNN.

    Returns:
        A tensor containing the kNN predictions


    """
    # compute cos similarity between each feature vector and feature bank ---> (B, N)
    sim_matrix = torch.mm(feature, feature_bank)
    # (B, K)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # (B, K)
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_labels = sim_labels.long()
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # (B*K, C)
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> (B, C)
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels[:, 0]
