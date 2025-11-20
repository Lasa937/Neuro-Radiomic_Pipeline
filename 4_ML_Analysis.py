#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:38:40 2024

@author: meyer_ai
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import os
import numpy as np

# 1. Load the CSV files
mean_hgg = pd.read_csv('/path/to/class1/mean/features/csv')
mean_lgg = pd.read_csv('/path/to/class2/mean/features/csv')
robust_hgg = pd.read_csv('/path/to/class1/robust/features/csv')
robust_lgg = pd.read_csv('/path/to//class2/robust/features/csv')

# 2. Identify the robust features (combine lists from HGG and LGG)
robust_features_hgg = set(robust_hgg.iloc[:, 0])
robust_features_lgg = set(robust_lgg.iloc[:, 0])
robust_features = list(robust_features_hgg.intersection(robust_features_lgg))

# 3. Prepare the dataset by merging the features for HGG and LGG
mean_hgg['Label'] = 1
mean_lgg['Label'] = 0

# Combine datasets
combined_data = pd.concat([mean_hgg, mean_lgg], axis=0)
combined_data = combined_data.drop(columns=['Patient_ID'])  # Drop the ID column

# Separate features and labels
X_all_features = combined_data.drop(columns=['Label'])
y = combined_data['Label']

# Filter for robust features
X_robust_features = X_all_features[robust_features]

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)

def evaluate_model_with_cv(X, y, model_name="RandomForest"):
    if model_name == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(random_state=42, use_label_encoder=False)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    feature_importance_data_shap = []
    all_test_probabilities = []
    all_test_predictions = []
    all_test_true = []

    for i, (train_val_index, test_index) in enumerate(outer_cv.split(X, y), 1):
        X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
        y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]

        pipeline = Pipeline(steps=[('sampling', smote), ('model', model)])
        pipeline.fit(X_train_val, y_train_val)
        
        test_predictions = pipeline.predict(X_test)
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]

        all_test_predictions.extend(test_predictions)
        all_test_probabilities.extend(test_probabilities)
        all_test_true.extend(y_test)

        fold_result = {
            'Fold': i,
            'Test Accuracy': accuracy_score(y_test, test_predictions),
            'Test Balanced Accuracy': balanced_accuracy_score(y_test, test_predictions),
            'Test Precision': precision_score(y_test, test_predictions),
            'Test Recall': recall_score(y_test, test_predictions),
            'Test F1-Score': f1_score(y_test, test_predictions),
            'Test AUC-ROC': roc_auc_score(y_test, test_probabilities)
        }
        
        fold_results.append(fold_result)

        # SHAP values for model interpretation
        explainer = shap.TreeExplainer(pipeline.named_steps['model'])
        shap_values = explainer.shap_values(X_train_val)
        
        # Handle the SHAP output based on its type
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 SHAP values for binary classification
        
        feature_importance_data_shap.append(pd.DataFrame({
            'Feature': X.columns,
            'Importance': abs(shap_values).mean(axis=0)
        }))

    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    
    # Calculate mean and standard deviation for each metric
    mean_results = results_df.mean().to_dict()
    std_results = results_df.std().to_dict()
    
    mean_results['Model'] = model_name
    std_results['Model'] = model_name + " (std)"
    
    shap_importance = pd.concat(feature_importance_data_shap).groupby('Feature').mean().sort_values(by='Importance', ascending=False).reset_index()
    
    return results_df, mean_results, std_results, shap_importance, all_test_true, all_test_predictions, all_test_probabilities

# Example Usage:
print("Evaluating Random Forest")
rf_results_df, rf_mean_results, rf_std_results, rf_shap_importance, rf_true, rf_preds, rf_probs = evaluate_model_with_cv(X_robust_features, y, model_name="RandomForest")

print("Evaluating XGBoost")
xgb_results_df, xgb_mean_results, xgb_std_results, xgb_shap_importance, xgb_true, xgb_preds, xgb_probs = evaluate_model_with_cv(X_robust_features, y, model_name="XGBoost")

def save_and_plot_results(results_df, mean_results, std_results, shap_importance_df, true_labels, predictions, probabilities, model_name, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results_df directly if it's not already a DataFrame
    if not isinstance(results_df, pd.DataFrame):
        results_df = pd.DataFrame([results_df])  # Convert results_dict to DataFrame
    
    # Save the results to an Excel file
    excel_path = os.path.join(output_dir, f'{model_name}_classification_results.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Fold Results', index=False)
        pd.DataFrame([mean_results]).to_excel(writer, sheet_name='Mean Results', index=False)
        pd.DataFrame([std_results]).to_excel(writer, sheet_name='Std Results', index=False)
        shap_importance_df.to_excel(writer, sheet_name='SHAP Feature Importance', index=False)

    # Create and save SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_importance_df['Importance'].values.reshape(-1, 1), shap_importance_df['Feature'].values, show=False, plot_type='bar')
    plt.title(f'SHAP Summary Plot for {model_name}')
    shap_plot_path = os.path.join(output_dir, f'shap_summary_plot_{model_name}.png')
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()

    # Add SHAP plot image to the Excel file
    workbook = load_workbook(excel_path)
    sheet = workbook['SHAP Feature Importance']
    img = Image(shap_plot_path)
    sheet.add_image(img, 'F1')
    workbook.save(excel_path)

    # Optionally, remove the temporary image file
    if os.path.exists(shap_plot_path):
        os.remove(shap_plot_path)

    # Plot and save the ROC curve
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_score(true_labels, probabilities):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(output_dir, f'roc_curve_{model_name}.png')
    plt.savefig(roc_curve_path, bbox_inches='tight')
    plt.close()

    # Plot and save the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    print(f'Results and plots saved for {model_name} in {output_dir}')

# Example usage:
output_dir = '/path/to/output/results/csv'

# Save and plot for Random Forest
save_and_plot_results(rf_results_df, rf_mean_results, rf_std_results, rf_shap_importance, rf_true, rf_preds, rf_probs, "RandomForest", output_dir)

# Save and plot for XGBoost
save_and_plot_results(xgb_results_df, xgb_mean_results, xgb_std_results, xgb_shap_importance, xgb_true, xgb_preds, xgb_probs, "XGBoost", output_dir)
