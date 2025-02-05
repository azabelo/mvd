from s3d import S3D
import torch

video_encoder = S3D('pretrained_models/s3d_dict.npy', 512)
video_encoder.load_state_dict(
            torch.load('pretrained_models/s3d_howto100m.pth'))

input_tensor = torch.rand(2, 3, 32, 224, 224)
output = video_encoder(input_tensor)['video_embedding']





# Print the output shape
print("Output shape:", output.shape)