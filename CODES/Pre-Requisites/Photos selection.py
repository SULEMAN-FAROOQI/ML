import os
import random
import shutil

# 1. Define your folder paths (change these to your actual paths)
source_folder = "C:\\Users\\sulem\\OneDrive\\Pictures\\UTKFace"
destination_folder = "C:\\Users\\sulem\\OneDrive\\Desktop\\Codes\\ML\\Datasets\\UTK Face\\train"
num_files_to_select = 10000

# 2. Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 3. Get a list of all files in the source folder
# (Optional: Filters for common image formats)
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
all_files = [
    f for f in os.listdir(source_folder) 
    if os.path.isfile(os.path.join(source_folder, f)) and f.lower().endswith(image_extensions)
]

# 4. Check if there are enough files to select from
if len(all_files) < num_files_to_select:
    print(f"Error: The folder only contains {len(all_files)} images, but you requested {num_files_to_select}.")
else:
    # 5. Randomly sample 10,000 unique files (random.sample never repeats an item)
    selected_files = random.sample(all_files, num_files_to_select)

    # 6. Copy the selected files to the new folder
    print(f"Starting to copy {num_files_to_select} files...")
    for index, file_name in enumerate(selected_files, 1):
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        
        shutil.copy(source_path, destination_path)
        
        # Show progress every 1,000 files
        if index % 1000 == 0:
            print(f"Copied {index}/{num_files_to_select} files...")

    print("Successfully completed!")
