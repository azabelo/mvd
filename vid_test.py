from s3d import S3D
import torch

video_encoder = S3D('pretrained_models/s3d_dict.npy', 512)
video_encoder.load_state_dict(
            torch.load('pretrained_models/s3d_howto100m.pth'))