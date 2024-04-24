#!/bin/bash

# Path to the notebook file
notebook="/home/local-admin/lightmycells/Label_free_cell_painting/notebooks/cellpainting_script_version_v4_tiling_uint.ipynb"

# Path to the directory containing subdirectories
main_dir="/media/local-admin/galaxy/lightmycells_storage/images"

# Counter for notebook number
folder_counter=1

# Iterate over subdirectories
for subdir in "$main_dir"/*; do
    # Check if it's a directory
    if [ -d "$subdir" ]; then
        subdir_with_slash="$subdir/"
        echo "Processing notebook #$folder_counter for directory: $subdir_with_slash"
        # Run the notebook with papermill, passing the directory path as a parameter
        output_notebook="/media/local-admin/lightmycells/Study_patches_s_3/tools/pytables/${folder_counter}_output.ipynb"
        papermill "$notebook" "$output_notebook" -p directory_path "$subdir_with_slash"
        
        # Increment notebook counter
        ((folder_counter++))
    fi
done

cd /home/local-admin/lightmycells/Label_free_cell_painting/notebooks
jupyter nbconvert --to script extend_pytables_v4_patches.ipynb

# Run the Python script
python extend_pytables_v4_patches.py > extend_logs.txt

cd /media/local-admin/lightmycells/Study_patches_s_3/tools
python own_patches_s_3_train_Unet.py > train_logs.txt

shutdown -h +5
