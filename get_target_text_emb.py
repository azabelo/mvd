import torch
import clip
from timm.models import create_model
import modeling_student
import modeling_teacher
import modeling_video_teacher
from collections import OrderedDict
from utils import load_state_dict

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

    class_names_str = "brush_hair clap draw_sword fall_floor handstand kick pick push run shoot_gun smoke sword turn cartwheel climb dribble fencing hit kick_ball pour pushup shake_hands sit somersault sword_exercise walk catch climb_stairs drink flic_flac hug kiss pullup ride_bike shoot_ball situp stand talk wave chew dive eat golf jump laugh punch ride_horse shoot_bow smile swing_baseball throw"

    # Convert the string to a list by splitting on spaces and then removing underscores
    class_names = class_names_str.split()
    # Sort the list alphabetically
    class_names.sort()

    prompts = []
    for name in class_names:
        for template in templates:
            prompts.append(template.format(name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    text_embs = clip.tokenize(["car", "a red car", "rowing"]).to(device)
    text_features = model.encode_text(text_embs)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features

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

        load_state_dict(video_teacher_model, checkpoint_model, prefix='')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_teacher_model.to(device)
    return video_teacher_model