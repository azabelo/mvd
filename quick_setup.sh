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

directory="hmdb51_mp4"  # Replace this with the path to your starting directory
output_csv="train.csv"  # Replace this with the desired output CSV file name

# Function to get the number of frames in a video file
get_frame_count() {
  # Use ffprobe to get the frame count from the video file
  frame_count=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$1")
  echo "$frame_count"
}

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Error: Directory '$directory' not found."
  exit 1
fi

# Create the output CSV file
touch "$output_csv"

# Function to list all files in a directory (recursively)
list_files() {
  local current_dir="$1"

  for file in "$current_dir"/*; do
    if [ -f "$file" ]; then
      # Print the relative path, video duration, and video label to the output CSV
      relative_path="${file#"$directory"}"
      frame_count=$(get_frame_count "$file")
      video_label=$(basename "$current_dir")
      echo "$directory$relative_path $frame_count $video_label" >> "$output_csv"
    elif [ -d "$file" ]; then
      # If it's a subdirectory, recursively call the function
      list_files "$file"
    fi
  done
}

# Call the function to list all files in the directory and its subdirectories
list_files "$directory"

echo "CSV file created: $output_csv"