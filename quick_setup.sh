#!/bin/bash

######## GET DEPENDENCIES #########

requirements_file="requirements.txt"

# Check if the requirements file exists
if [ ! -f "$requirements_file" ]; then
  echo "Error: File '$requirements_file' not found."
  exit 1
fi

# Read the requirements file line by line and install each module using pip
while IFS= read -r module; do
  if [[ "$module" != \#* ]]; then
    # Skip lines starting with '#' (comments) in the requirements file
    pip install "$module"
  fi
done < "$requirements_file"

echo "All modules from '$requirements_file' have been installed."

######## DOWNLOAD DATASET #########

gdown https://drive.google.com/uc?id=1SeLNhVD92qqE0MaQAZOEIyF5uq3a2wKS --output hmdb51_mp4.zip
sudo apt-get install unzip
unzip hmdb51_mp4.zip



######## MAKE CSV FILE ############

gdown https://drive.google.com/uc?id=1SERyaCXqPfzhCI4z6WG0UA9m_jdqBSrb --output train.csv

gdown https://drive.google.com/uc?id=1ae_lhZTVrwIfmwyzrP_t2aofN9rPsjtk --output finetune_splits/train.csv
gdown https://drive.google.com/uc?id=1ae_lhZTVrwIfmwyzrP_t2aofN9rPsjtk --output finetune_splits/train.csv
gdown https://drive.google.com/uc?id=1ae_lhZTVrwIfmwyzrP_t2aofN9rPsjtk --output finetune_splits/train.csv

######## GET IMAGE AND VIDEO TEACHERS #########

#default MAE
curl https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth --output image_teacher.pth
#CLIP
gdown https://drive.google.com/uc?id=1x9svwSPTXHVe21mF--q4IbMC-yGFOXFR --output clip_model.pth
gdown 'https://drive.google.com/u/0/uc?id=1tEhLyskjb755TJ65ptsrafUG2llSwQE1&amp;export=download&amp;confirm=t&amp;uuid=63e3a20f-2e32-4603-bc52-13a154ead88c&amp;at=ALt4Tm0vJDg8yrew90Qs81X3co6l:1691104865099&confirm=t&uuid=319d04ee-2975-41b8-b28e-2118e9b41167&at=ALt4Tm03Kkqy082aSFi3NPa54Un3:1691106288399' --output video_teacher.pth

## need this to run pretrain.sh (you'll need to click yes)
sudo apt-get install libgl1-mesa-glx
