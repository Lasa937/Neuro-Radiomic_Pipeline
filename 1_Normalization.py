import os
import numpy as np
import nibabel as nib
from scipy import stats

def normalize_t1(t1_path, seg_path, output_path):
    # Load T1 and segmentation images
    t1_img = nib.load(t1_path)
    t1_data = t1_img.get_fdata()
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    # Create mask for brainstem (labels 7, 8, 46, 47)
    brainstem_mask = np.isin(seg_data, [7, 8, 46, 47])

    # Extract brainstem intensities
    brainstem_intensities = t1_data[brainstem_mask]

    # Calculate median and IQR of brainstem intensities
    median = np.median(brainstem_intensities)
    iqr_val = stats.iqr(brainstem_intensities)

    # Check for IQR zero to avoid division by zero
    if iqr_val == 0:
        print(f"IQR is 0 for {os.path.basename(t1_path)}, normalization skipped.")
        return

    # Normalize T1 data
    normalized_data = (t1_data - median) / iqr_val

    # Create and save normalized image
    normalized_img = nib.Nifti1Image(normalized_data, t1_img.affine, t1_img.header)
    nib.save(normalized_img, output_path)
    print(f"Processed and saved: {os.path.basename(t1_path)}")

def process_all_images(t1_folder, seg_folder, output_folder):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(t1_folder):
        if filename.endswith('.nii.gz'):
            t1_path = os.path.join(t1_folder, filename)
            seg_path = os.path.join(seg_folder, filename)
            
            if os.path.exists(seg_path):
                output_path = os.path.join(output_folder, f"{filename}")
                normalize_t1(t1_path, seg_path, output_path)
            else:
                print(f"Skipping {filename}: No segmentation found")

# Set up paths
t1_folder = r"path/to/MR/folder"
seg_folder = r"path/to/healthy/segmentation/folder"
output_folder = r"path/to/MR/normalized/output/folder"

# Process all images
process_all_images(t1_folder, seg_folder, output_folder)