import os
import shutil
from project_paths import results_dir

def find_and_delete_unmatched_test_folders(base_dir):
    # Get all test folders
    test_folders = [f for f in os.listdir(base_dir) if f.startswith("test_")]
    
    for test_folder in test_folders:
        # Build the corresponding training folder path
        corresponding_train_folder = test_folder.replace("test_", "exp_")
        
        test_folder_path = os.path.join(base_dir, test_folder)
        train_folder_path = os.path.join(base_dir, corresponding_train_folder)
        
        # Delete the test folder if the corresponding training folder does not exist
        if not os.path.exists(train_folder_path):
            print(f"[INFO] Training folder for test folder {test_folder} does not exist. Deleting {test_folder}.")
            shutil.rmtree(test_folder_path)
        else:
            print(f"[INFO] Found corresponding training folder for {test_folder}. No action needed.")
    
    print("[INFO] Folder check and cleanup complete.")

# Pass in the root directory path
base_directory = str(results_dir("results/training_runs"))
find_and_delete_unmatched_test_folders(base_directory)
