#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unsupervised Learning Analysis for Amazon Food Reviews

This script implements KMeans clustering on Amazon food reviews data
to discover natural groupings based on sentiment and other features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def load_data(data_path, sample_size=100000):
    """Load the dataset with optional sampling"""
    print(f"Loading data from {data_path}...")
    data_file = pd.read_csv(data_path)
    print(f"Original dataset shape: {data_file.shape}")
    
    if sample_size and len(data_file) > sample_size:
        data_file = data_file.sample(n=sample_size, random_state=42)
        print(f"Sampled dataset shape: {data_file.shape}")
    
    return data_file

def prepare_features(data_file):
    """Prepare features for clustering"""
    # Numerical features
    num_features = ['HelpfulnessNumerator', 'HelpfulnessDenominator',
                   'text_length', 'summary_length', 'helpful_ratio']
    
    # Sentiment features
    sentiment_features = ['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']
    
    # All features for clustering
    all_features = num_features + sentiment_features
    
    # Select features and handle missing values
    X = data_file[all_features].copy()
    X.fillna(0, inplace=True)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, all_features  

def find_optimal_clusters(X_scaled, max_clusters=10):
    """Find the optimal number of clusters using the elbow method and silhouette score"""
    print("Finding optimal number of clusters...")
    
    inertia_values = []
    silhouette_values = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia_values.append(kmeans.inertia_)
        silhouette_values.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot elbow method
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertia_values, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_values, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find the optimal k based on silhouette score
    optimal_k = np.argmax(silhouette_values) + 2  # +2 because we started from k=2
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
    
    return optimal_k

def perform_clustering(X_scaled, n_clusters, feature_names, data_file):
    """Perform KMeans clustering and visualize results"""
    print(f"\nPerforming clustering with {n_clusters} clusters...")
    
    # Fit KMeans with optimized parameters
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42,
        n_init=5,  # Reduced from 10 for computation 
        max_iter=200,
        algorithm='elkan'
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to original data
    data_file_copy = data_file.copy()
    data_file_copy['Cluster'] = cluster_labels
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create visualization plots
    plt.figure(figsize=(15, 10))
    
    # Scatter plot of clusters
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Clusters Visualization (PCA)')
    
    # Distribution of clusters
    plt.subplot(2, 2, 2)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Cluster Size Distribution')
    
    # Cluster centers analysis
    plt.subplot(2, 2, 3)
    cluster_centers = kmeans.cluster_centers_
    im = plt.imshow(cluster_centers.T, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.xlabel('Cluster')
    plt.ylabel('Feature Index')
    plt.title('Cluster Centers Heatmap')
    
    # Silhouette score by cluster
    plt.subplot(2, 2, 4)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    plt.text(0.5, 0.5, f'Average Silhouette Score: {silhouette_avg:.3f}', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=14)
    plt.title('Clustering Quality')
    plt.axis('off')
    
    # Save results
    plt.tight_layout()
    plt.savefig('clusters_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create cluster analysis summary
    cluster_summary = data_file_copy.groupby('Cluster').agg({
        'Score': ['mean', 'std', 'count'],
        'sentiment_compound': ['mean', 'std'],
        'text_length': ['mean', 'std']
    }).round(3)
    
    return data_file_copy, cluster_summary

def main():
    parser = argparse.ArgumentParser(description='Unsupervised Learning Analysis for Amazon Food Reviews')
    parser.add_argument('--data_path', type=str, default='cleaned_data.csv',
                        help='Path to the data file (default: cleaned_data.csv)')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters (default: auto-determined)')
    parser.add_argument('--max_clusters', type=int, default=10,
                        help='Maximum number of clusters to consider (default: 10)')
    parser.add_argument('--output', type=str, default='clustered_data.csv',
                        help='Path to save the clustered data (default: clustered_data.csv)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        data_file = load_data(args.data_path)
        
        # Prepare features - update this line
        X_scaled, feature_names = prepare_features(data_file)
        
        # Find optimal number of clusters if not specified
        n_clusters = args.n_clusters
        if n_clusters is None:
            n_clusters = find_optimal_clusters(X_scaled, args.max_clusters)
        
        # Perform clustering - update this line
        clustered_data, cluster_analysis = perform_clustering(X_scaled, n_clusters, feature_names, data_file)
        
        # Save clustered data
        clustered_data.to_csv(args.output, index=False)
        print(f"\nClustered data saved to '{args.output}'")
        
        # Save cluster analysis
        cluster_analysis.to_csv('cluster_analysis.csv')
        print("Cluster analysis saved to 'cluster_analysis.csv'")
        print("\nVisualization plots saved:")
        print("- optimal_clusters.png")
        print("- clusters_visualization.png")
        
        print("\nUnsupervised analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in unsupervised analysis: {e}")

if __name__ == "__main__":
    main()