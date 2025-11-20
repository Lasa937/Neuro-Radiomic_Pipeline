#%%
import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
from radiomics import featureextractor
import ast

#%%
# Function to calculate ICC
def calculate_icc(data):
    mean_square_between = np.var(np.mean(data, axis=1), ddof=1)
    mean_square_within = np.mean(np.var(data, axis=1, ddof=1))
    n_raters = data.shape[1]
    icc = (mean_square_between - mean_square_within) / (mean_square_between + (n_raters - 1) * mean_square_within)
    return icc

# Read pathology data
pathology_file = "path/to/label/csv/"
pathology_df = pd.read_csv(pathology_file)
pathology_dict = dict(zip(pathology_df["BraTS_2020_subject_ID"], pathology_df["Grade"]))

# Define directories
t1_dir = r"path/to/MR/folder"
seg_dir = r"path/to/segmentation/modified/output/folder"

# List all T1 files
t1_files = [f for f in os.listdir(t1_dir) if f.endswith('.nii.gz')]

# Initialize a feature extractor with 128 gray levels
extractor = featureextractor.RadiomicsFeatureExtractor(binCount=128)

# Initialize dictionaries to store features for HGG and LGG
hgg_features = {}
lgg_features = {}

#%%
for t1_file in t1_files:
    base_name = t1_file.replace('_t1.nii.gz', '')
    t1_path = os.path.join(t1_dir, t1_file)
    
    # Determine the pathology of the patient
    pathology = pathology_dict.get(base_name)
    
    if pathology is None:
        continue  # Skip if pathology data is missing
    
    # Load the T1 image
    t1_image = sitk.ReadImage(t1_path)
    
    # Define the segmentation paths
    seg_paths = [
        os.path.join(seg_dir, f"{base_name}_t1_human_error_seg.nii.gz"),
        os.path.join(seg_dir, f"{base_name}_t1_combined_seg.nii.gz"),
        os.path.join(seg_dir, f"{base_name}_t1_random_zero_seg.nii.gz")
    ]
    
    # Store the features for each segmentation
    features = []
    for seg_path in seg_paths:
        seg_image = sitk.ReadImage(seg_path)
        result = extractor.execute(t1_image, seg_image)
        features.append({key: float(value) for key, value in result.items() if "diagnostics" not in key})
    
    # Save the features based on pathology
    if pathology == "HGG":
        hgg_features[base_name] = features
    elif pathology == "LGG":
        lgg_features[base_name] = features
        
#%%
# Save the features to CSV files
hgg_features_df = pd.DataFrame.from_dict(hgg_features, orient='index')
lgg_features_df = pd.DataFrame.from_dict(lgg_features, orient='index')
hgg_features_df.to_csv('/path/to/output/class1/label/csv')
lgg_features_df.to_csv('/path/to/output/class2/label/csv')

#%%
# Convert features to DataFrame for ICC analysis
def extract_icc_data(feature_dict):
    feature_names = list(feature_dict[list(feature_dict.keys())[0]][0].keys())
    iccs = []

    for feature in feature_names:
        # Collect values across all segmentations for each feature
        feature_values = [feature_dict[base_name][i][feature] for base_name in feature_dict for i in range(3)]
        # Reshape to have (number of images, number of segmentations)
        feature_matrix = np.array(feature_values).reshape(-1, 3)
        # Perform ICC analysis
        icc = calculate_icc(feature_matrix)
        iccs.append({'Feature': feature, 'ICC': icc})

    return pd.DataFrame(iccs)

# Calculate ICCs for HGG and LGG
icc_hgg_df = extract_icc_data(hgg_features)
icc_lgg_df = extract_icc_data(lgg_features)

# Save ICC results
icc_hgg_df.to_csv('/path/to/output/class1/ICC/csv', index=False)
icc_lgg_df.to_csv('/path/to/output/class2/ICC/csv', index=False)

# Filter robust features based on ICC threshold
robust_hgg_features = icc_hgg_df[icc_hgg_df['ICC'] >= 0.75]
robust_lgg_features = icc_lgg_df[icc_lgg_df['ICC'] >= 0.75]

# Save the robust features
robust_hgg_features.to_csv('/path/to/output/class1/robust/csv', index=False)
robust_lgg_features.to_csv('/path/to/output/class2/robust/csv', index=False)

#%%

def calculate_mean_radiomic_features(input_csv, output_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Convert the JSON-like strings into dictionaries
    df['features_0'] = df['0'].apply(ast.literal_eval)
    df['features_1'] = df['1'].apply(ast.literal_eval)
    df['features_2'] = df['2'].apply(ast.literal_eval)

    # Extract the list of features from one of the columns
    sample_features = df['features_0'].iloc[0].keys()

    # Create a new DataFrame to store the mean values
    mean_features_df = pd.DataFrame()

    # Set the index as the patient ID
    mean_features_df['Patient_ID'] = df['Unnamed: 0']

    # Calculate the mean for each feature across the three segmentations
    for feature in sample_features:
        mean_features_df[feature] = df.apply(
            lambda row: (row['features_0'][feature] + row['features_1'][feature] + row['features_2'][feature]) / 3, axis=1
        )

    # Save the resulting DataFrame to a new CSV file
    mean_features_df.to_csv(output_csv, index=False)

# Example usage
input_csv = '/path/to/input/class1/features/csv'
output_csv = '/path/to/output/class1/mean/features/csv'
calculate_mean_radiomic_features(input_csv, output_csv)

input_csv = '/path/to/input/class2/features/csv'
output_csv = '/path/to/output/class2/mean/features/csv'
calculate_mean_radiomic_features(input_csv, output_csv)