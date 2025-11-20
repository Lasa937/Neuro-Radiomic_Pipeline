#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:34:45 2024

@author: meyer_ai
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to load the data
def load_data(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    return pd.read_excel(file_path, sheet_name='Fold Results', engine='openpyxl')

# Function to plot the comparison between all features and robust features
def plot_comparison(all_features_df, robust_features_df, model_name):
    metrics = ['Test Accuracy', 'Test Balanced Accuracy', 'Test Precision', 
               'Test Recall', 'Test F1-Score', 'Test AUC-ROC']
    
    # Create a new DataFrame to hold the combined data
    combined_df = pd.DataFrame()

    for metric in metrics:
        all_features_metric = all_features_df[metric]
        robust_features_metric = robust_features_df[metric]
        
        temp_df = pd.DataFrame({
            'Metric': [metric] * len(all_features_metric) * 2,
            'Value': pd.concat([all_features_metric, robust_features_metric]),
            'Feature Set': ['All Features'] * len(all_features_metric) + ['Robust Features'] * len(robust_features_metric)
        })
        
        combined_df = pd.concat([combined_df, temp_df], axis=0)

    plt.figure(figsize=(14, 8))
    
    sns.pointplot(data=combined_df, x='Metric', y='Value', hue='Feature Set', linestyles='', dodge=True, markers=["o", "s"], capsize=.1, palette="Set2")
    
    plt.title(f'{model_name} - Comparison of Metrics: All Features vs Robust Features', fontsize=16)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.legend(title='Feature Set')
    plt.show()

# Define the file and folder paths
base_path = "/path/to/principal/folder"
all_features_folder = "folder_name_all"
robust_features_folder = "folder_name_all"
file_name = "file_name_results"

# Load data for Random Forest
rf_all_features_df = load_data(os.path.join(base_path, all_features_folder), file_name)
rf_robust_features_df = load_data(os.path.join(base_path, robust_features_folder), file_name)

# Plot comparison for Random Forest
#plot_comparison(rf_all_features_df, rf_robust_features_df, "Random Forest")
plot_comparison(rf_all_features_df, rf_robust_features_df, "XGBoost")
