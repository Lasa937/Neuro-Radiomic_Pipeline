import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.morphology import remove_small_objects, remove_small_holes

def randomly_modify_boundary(mask, percent_to_modify, dilation_size=3):
    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(mask, structure=struct, iterations=dilation_size)
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=dilation_size)
    boundary = dilated ^ eroded
    boundary_indices = np.argwhere(boundary)
    num_to_modify = int(percent_to_modify * len(boundary_indices))
    chosen_indices = np.random.choice(len(boundary_indices), num_to_modify, replace=False)
    chosen_boundary_points = boundary_indices[chosen_indices]
    for point in chosen_boundary_points:
        if np.random.rand() > 0.5:
            mask[tuple(point)] = 1  # Enlarge (dilate)
        else:
            mask[tuple(point)] = 0  # Shrink (erode)
    return mask

def set_random_pixels_to_zero(mask, percent_to_zero):
    mask = mask.astype(bool)  # Ensure mask is boolean
    total_pixels = np.sum(mask)
    num_to_zero = int(percent_to_zero * total_pixels)
    mask_indices = np.argwhere(mask)
    chosen_indices = np.random.choice(len(mask_indices), num_to_zero, replace=False)
    mask_copy = mask.copy()
    mask_copy[tuple(mask_indices[chosen_indices].T)] = False
    return mask_copy

def simulate_human_error(mask, percent_to_modify, dilation_size=3):
    modified_mask = randomly_modify_boundary(mask.copy(), percent_to_modify, dilation_size)
    labeled, num_features = ndimage.label(modified_mask)
    largest_component = (labeled == (np.bincount(labeled.ravel())[1:].argmax() + 1))
    return remove_small_holes(largest_component)

def process_images(t1_folder, seg_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(t1_folder):
        if filename.endswith('.nii.gz'):
            t1_path = os.path.join(t1_folder, filename)
            seg_path = os.path.join(seg_folder, filename.replace('_t1.nii.gz', '_seg.nii.gz'))

            if not os.path.exists(seg_path):
                print(f"Segmentation not found for {filename}")
                continue

            # Load the segmentation
            seg_img = nib.load(seg_path)
            seg_data = seg_img.get_fdata()

            # Combine labels 1 and 4 into a single label, let's say 1
            combined_seg = np.zeros_like(seg_data)
            combined_seg[np.isin(seg_data, [1, 4])] = 1

            # Save the combined segmentation as the new original segmentation
            combined_seg_img = nib.Nifti1Image(combined_seg, seg_img.affine, seg_img.header)
            nib.save(combined_seg_img, os.path.join(output_folder, f"{filename.replace('.nii.gz', '_combined_seg.nii.gz')}"))

            # 2. Create segmentation with random pixels set to zero
            random_zero_seg = combined_seg.copy()
            modified_label_mask = set_random_pixels_to_zero(combined_seg, 0.3)
            random_zero_img = nib.Nifti1Image(modified_label_mask, seg_img.affine, seg_img.header)
            nib.save(random_zero_img, os.path.join(output_folder, f"{filename.replace('.nii.gz', '_random_zero_seg.nii.gz')}"))

            # 3. Simulate human error in segmentation
            human_error_seg = combined_seg.copy()
            modified_label_mask = simulate_human_error(combined_seg, 0.5, dilation_size=5)
            human_error_img = nib.Nifti1Image(modified_label_mask, seg_img.affine, seg_img.header)
            nib.save(human_error_img, os.path.join(output_folder, f"{filename.replace('.nii.gz', '_human_error_seg.nii.gz')}"))

            print(f"Processed {filename}")

if __name__ == "__main__":
    t1_folder = r"path/to/MR/normalized/folder" #produced in prvious script
    seg_folder = r"path/to/pathology/segmentation/folder"
    output_folder = r"path/to/segmentation/modified/output/folder"

    process_images(t1_folder, seg_folder, output_folder)