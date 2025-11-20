# Medical Image Analysis Pipeline for Brain Tumor Classification

## Overview
This repository contains a comprehensive pipeline for medical image analysis and machine learning classification of brain tumor types (Class1 and Class2). The pipeline can be used on multiple MRI sequences, extracts radiomic features, evaluates feature robustness, and builds predictive models.

## Pipeline Structure

The analysis consists of 5 sequential Python scripts:

### 1. `1_Normalization.py`
- **Purpose**: Normalizes MRI images using brainstem intensity values
- **Input**: Raw MRI sequences (T1, T1CE, T2, FLAIR) and corresponding segmentation files
- **Output**: Normalized MRI sequences
- **Method**: Uses median and interquartile range (IQR) of brainstem intensities for normalization
- **Compatibility**: Works with all standard MRI sequences

### 2. `2_Seg_Perturbation.py`
- **Purpose**: Generates modified segmentation masks to simulate real-world variations
- **Input**: Normalized MRI sequences and original segmentations
- **Output**: Three types of modified segmentations for each MRI sequence:
  - Combined segmentation (merging labels)
  - Random zero-pixel segmentation
  - Human error simulation segmentation
- **Method**: Applies morphological operations and random modifications
- **Note**: Processes all available MRI sequences in the input directory

### 3. `3_Features_Extraction.py`
- **Purpose**: Extracts radiomic features from multiple MRI sequences and evaluates their robustness
- **Input**: Normalized MRI sequences and modified segmentations
- **Output**: 
  - Extracted features for Class1 and Class2 from each MRI sequence
  - ICC (Intraclass Correlation Coefficient) analysis per sequence type
  - Robust feature selection (ICC ≥ 0.75) for each sequence
  - Mean feature values across segmentations
- **Method**: Uses PyRadiomics library for feature extraction across all MRI sequences

### 4. `4_ML_Analysis.py`
- **Purpose**: Machine learning classification and model interpretation using features from multiple MRI sequences
- **Input**: Mean features and robust feature lists from all available sequences
- **Output**: 
  - Cross-validated performance metrics
  - SHAP feature importance analysis
  - ROC curves and confusion matrices
  - Excel reports with results
- **Models**: Random Forest and XGBoost classifiers
- **Method**: 5-fold stratified cross-validation with SMOTE for class balancing
- **Feature Integration**: Combines features from all MRI sequences for comprehensive analysis

### 5. `5_plots.py`
- **Purpose**: Comparative visualization of model performance across feature sets
- **Input**: Results from ML analysis using different MRI sequences
- **Output**: Comparison plots between all features vs. robust features
- **Metrics**: Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Usage Instructions

### Prerequisites
- Python 3.7+
- Required libraries: 
  - SimpleITK, PyRadiomics, nibabel, scipy, scikit-learn, imbalanced-learn, xgboost, SHAP, pandas, numpy, matplotlib, seaborn, openpyxl

### Supported MRI Sequences
The pipeline is designed to work with all standard MRI sequences:
- **T1-weighted (T1)**
- **T1-weighted contrast-enhanced (T1CE)**
- **T2-weighted (T2)**
- **Fluid-attenuated inversion recovery (FLAIR)**
- **Other MRI sequences** (with appropriate file naming)

### Step-by-Step Execution

1. **Normalize MRI Sequences**:
   ```bash
   python 1_Normalization.py
   ```
   - Update file paths for your MRI sequences and segmentation directories
   - Script processes all MRI sequences found in the input directory

2. **Generate Modified Segmentations**:
   ```bash
   python 2_Seg_Perturbation.py
   ```
   - Uses normalized images from Step 1 for all available sequences
   - Generates modified segmentations for each MRI sequence type

3. **Extract and Analyze Features**:
   ```bash
   python 3_Features_Extraction.py
   ```
   - Processes all three segmentation types for each MRI sequence
   - Identifies robust features using ICC analysis per sequence type
   - Can be run separately for different MRI sequences or combined

4. **Run Machine Learning Analysis**:
   ```bash
   python 4_ML_Analysis.py
   ```
   - Trains and evaluates both Random Forest and XGBoost models
   - Can analyze features from individual sequences or combined multi-sequence features
   - Generates comprehensive performance reports

5. **Generate Comparative Plots**:
   ```bash
   python 5_plots.py
   ```
   - Visualizes performance differences between feature sets
   - Can compare performance across different MRI sequences

## File Structure Requirements

Before running the pipeline, ensure you have the following directory structure:

```
project_root/
├── raw_data/
│   ├── MRI_sequences/
│   │   ├── T1/            # T1-weighted images
│   │   ├── T1CE/          # T1 contrast-enhanced images
│   │   ├── T2/            # T2-weighted images
│   │   ├── FLAIR/         # FLAIR images
│   │   └── [other_sequences]/
│   ├── segmentations/      # Original segmentation masks
│   └── pathology_labels/   # Class labels (Class1/Class2)
├── normalized_images/      # Output from 1_Normalization.py
│   ├── T1/
│   ├── T1CE/
│   ├── T2/
│   └── FLAIR/
├── modified_segmentations/ # Output from 2_Seg_Perturbation.py
│   ├── T1/
│   ├── T1CE/
│   ├── T2/
│   └── FLAIR/
└── results/               # Output from 3, 4, 5 scripts
    ├── features/
    │   ├── T1/
    │   ├── T1CE/
    │   ├── T2/
    │   └── FLAIR/
    ├── ICC_analysis/
    └── ML_results/
```

## Configuration

Update the following paths in each script before execution:
- Input directories for MRI sequences and segmentations
- Output directories for processed files and results
- Pathology label file paths
- Specify which MRI sequences to include in the analysis

## Multi-Sequence Analysis Options

The pipeline supports several analysis approaches:

1. **Single Sequence Analysis**: Run the entire pipeline for one MRI sequence type
2. **Multi-Sequence Combined**: Extract features from all sequences and combine them for analysis
3. **Sequence Comparison**: Compare performance across different MRI sequences

## Key Features

- **Multi-Sequence Support**: Compatible with all standard MRI sequences
- **Robust Feature Selection**: Uses ICC analysis to identify reproducible features per sequence
- **Class Imbalance Handling**: Implements SMOTE for balanced training
- **Comprehensive Evaluation**: Multiple metrics and cross-validation
- **Model Interpretability**: SHAP analysis for feature importance across sequences
- **Segmentation Variability**: Accounts for real-world segmentation inconsistencies

## Output

The pipeline generates:
- Normalized medical images for all MRI sequences
- Multiple segmentation variants for each sequence
- Extracted radiomic features per sequence type
- ICC analysis results for each MRI sequence
- Machine learning model performance reports
- Feature importance rankings across sequences
- Comparative visualizations

For questions or issues, please check the script comments for detailed parameter explanations and ensure all file paths are correctly configured for your specific MRI sequences and environment.
