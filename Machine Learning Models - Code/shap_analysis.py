#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHAP Analysis for Amazon Food Reviews Sentiment Model

This script adds model interpretability to the Amazon Food Reviews Sentiment Analysis project
using SHAP (SHapley Additive exPlanations) to explain model predictions.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import argparse
import os

def load_data(data_path):
    """Load the dataset"""
    print(f"Loading data from {data_path}...")
    data_file = pd.read_csv(data_path)
    print(f"Dataset shape: {data_file.shape}")
    return data_file

def prepare_features(data_file):
    """Prepare features for SHAP analysis"""
    # Numerical features
    num_features = ['HelpfulnessNumerator', 'HelpfulnessDenominator',
                   'text_length', 'summary_length', 'helpful_ratio']
    
    # Sentiment features
    sentiment_features = ['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']
    
    # All features for this simplified model
    all_features = num_features + sentiment_features
    
    # Split the data into training and testing sets
    X = data_file[all_features]
    y = data_file['Score']  # Using Score directly as target
    
    return X, y, all_features

def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    return model

def explain_model_global(model, X, feature_names, max_display=20, plot_type='bar'):
    """Generate global SHAP explanations for the model"""
    print("Generating global SHAP explanations...")
    
    # Extract the actual model from the pipeline
    model_component = model.named_steps['model']
    
    # Apply the preprocessor to the data
    X_processed = model.named_steps['preprocessor'].transform(X)
    
    # Create a SHAP explainer appropriate for the model type
    if hasattr(model_component, 'feature_importances_'):
        # For tree-based models
        explainer = shap.TreeExplainer(model_component)
    else:
        # For other models
        explainer = shap.Explainer(model_component, X_processed)
    
    # Calculate SHAP values
    shap_values = explainer(X_processed)
    
    # Plot global feature importance
    plt.figure(figsize=(12, 8))
    if plot_type == 'bar':
        shap.plots.bar(shap_values, max_display=max_display, show=False)
    elif plot_type == 'beeswarm':
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig('shap_global_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Global SHAP plot saved to 'shap_global_importance.png'")
    
    return shap_values, explainer

def explain_model_local(model, explainer, X, feature_names, sample_indices=None, num_samples=5):
    """Generate local SHAP explanations for specific samples"""
    print("Generating local SHAP explanations...")
    
    # If no sample indices provided, randomly select some
    if sample_indices is None:
        sample_indices = np.random.choice(len(X), size=min(num_samples, len(X)), replace=False)
    
    # Get the samples
    X_samples = X.iloc[sample_indices]
    
    # Apply preprocessing
    X_processed = model.named_steps['preprocessor'].transform(X_samples)
    
    # Calculate SHAP values for the samples
    shap_values = explainer(X_processed)
    
    # Plot local explanations for each sample
    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[i], max_display=20, show=False)
        plt.title(f"Sample {idx} Explanation")
        plt.tight_layout()
        plt.savefig(f'shap_local_explanation_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Local SHAP plot for sample {idx} saved to 'shap_local_explanation_{idx}.png'")

def main():
    parser = argparse.ArgumentParser(description='SHAP Analysis for Amazon Food Reviews Sentiment Model')
    parser.add_argument('--data_path', type=str, default='cleaned_data.csv',
                        help='Path to the data file (default: cleaned_data.csv)')
    parser.add_argument('--model_path', type=str, default='amazon_reviews_sentiment_model.pkl',
                        help='Path to the trained model file (default: amazon_reviews_sentiment_model.pkl)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples for local explanations (default: 5)')
    parser.add_argument('--max_display', type=int, default=20,
                        help='Maximum number of features to display in plots (default: 20)')
    
    args = parser.parse_args()
    
    try:
        # Load data and model
        data_file = load_data(args.data_path)
        X, y, feature_names = prepare_features(data_file)
        model = load_model(args.model_path)
        
        # Generate global explanations
        shap_values, explainer = explain_model_global(model, X, feature_names, args.max_display)
        
        # Generate local explanations for random samples
        explain_model_local(model, explainer, X, feature_names, num_samples=args.num_samples)
        
        print("\nSHAP analysis completed successfully!")
        print("Global feature importance plot saved to 'shap_global_importance.png'")
        print(f"Local explanation plots saved for {args.num_samples} samples.")
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")

if __name__ == "__main__":
    main()