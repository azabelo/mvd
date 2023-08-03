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

# Specify the URL of the file to download
curl https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar --output hmdb51_dataset.rar

# Install unrar if not already installed
if ! command -v unrar &> /dev/null; then
    echo "Installing unrar..."
    sudo apt-get install unrar
fi

#unzip big file
mkdir hmdb51_unrared
mv hmdb51_dataset.rar hmdb51_unrared/hmdb51_dataset.rar
cd hmdb51_unrared
unrar x hmdb51_dataset.rar


######### UNRAR #########

# Get the current directory
directory=$(pwd)
# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Error: Directory '$directory' not found."
  exit 1
fi

# Change to the target directory
cd "$directory"

# Extract .rar files
rar_files=$(ls -p | grep -v / | grep -e "\.rar$")
for rar_file in $rar_files; do
    unrar x "$rar_file"
    rm "$rar_file"
done

echo "Extraction complete. .rar files in $directory have been unrar'd."
cd ..

########## CONVERT TO MP4 ##########

# Install ffmpeg if not already installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    # Use the appropriate package manager based on your system (apt-get for Ubuntu/Debian)
    # Replace 'apt-get' with 'brew' for macOS using Homebrew or 'yum' for some Linux distributions
    sudo apt-get install ffmpeg
fi

# Function to convert .avi to .mp4 using ffmpeg
convert_avi_to_mp4() {
  local input_file="$1"
  local output_file="$2"

  ffmpeg -i "$input_file" "$output_file"
}

# Main script
input_directory="hmdb51_unrared"  # Replace this with the path to your input directory
output_directory="hmdb51_mp4"  # Replace this with the path to the output directory

# Check if input directory exists
if [ ! -d "$input_directory" ]; then
  echo "Error: Input directory '$input_directory' not found."
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_directory"

# Find all subdirectories in the input directory
subdirectories=$(find "$input_directory" -type d -mindepth 1)

# Loop through each subdirectory and convert .avi files to .mp4
for subdirectory in $subdirectories; do
  # Create corresponding subdirectory in the output directory
  relative_path="${subdirectory#"$input_directory"/}"
  output_subdirectory="$output_directory/$relative_path"
  mkdir -p "$output_subdirectory"

  # Find all .avi files in the current subdirectory
  avi_files=$(find "$subdirectory" -type f -name "*.avi")

  # Convert .avi to .mp4 for each file found
  for avi_file in $avi_files; do
    # Get the relative path of the input file within the subdirectory
    relative_file_path="${avi_file#"$subdirectory"/}"
    # Construct the corresponding output path within the output subdirectory
    output_path="$output_subdirectory/${relative_file_path%.avi}.mp4"

    # Convert the .avi file to .mp4 using ffmpeg
    convert_avi_to_mp4 "$avi_file" "$output_path"
  done
done

echo "Conversion complete. .avi files in $input_directory and its subdirectories have been converted to .mp4 files in $output_directory."

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


