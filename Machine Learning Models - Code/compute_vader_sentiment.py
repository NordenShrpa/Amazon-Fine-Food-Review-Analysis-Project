#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute VADER Sentiment for Amazon Food Reviews Dataset

This script efficiently computes VADER sentiment scores for the entire Amazon Food Reviews dataset
using vectorized operations with pandas apply method. The results are saved back to the original
CSV file with additional sentiment score columns.
"""

import pandas as pd
import numpy as np
import nltk
import argparse
import time
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    print("Warning: Could not download NLTK vader_lexicon resource")

def compute_vader_sentiment(data_path, output_path=None):
    """
    Compute VADER sentiment scores for the entire dataset using vectorized operations
    
    Args:
        data_path (str): Path to the Amazon Food Reviews CSV file
        output_path (str, optional): Path to save the updated CSV file. If None, overwrites the input file.
    
    Returns:
        pd.DataFrame: DataFrame with added sentiment score columns
    """
    print(f"Loading data from {data_path}...")
    start_time = time.time()
    
    # Load the dataset
    data_file = pd.read_csv(data_path)
    print(f"Dataset shape: {data_file.shape}")
    
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Vectorized computation of sentiment scores
    print("Computing VADER sentiment scores for all reviews (this may take a while)...")
    
    # Apply sentiment analysis to all rows in a vectorized manner
    # First ensure Text column has no NaN values
    data_file['Text'] = data_file['Text'].fillna('')
    
    # Use apply with lambda function to compute sentiment for each review
    sentiments = data_file['Text'].apply(lambda t: sid.polarity_scores(t))
    
    # Convert the series of dictionaries to a DataFrame
    sentiment_df = pd.DataFrame(sentiments.tolist(), index=data_file.index)
    
    # Add sentiment columns to the original DataFrame
    data_file['sentiment_neg'] = sentiment_df['neg']
    data_file['sentiment_neu'] = sentiment_df['neu']
    data_file['sentiment_pos'] = sentiment_df['pos']
    data_file['sentiment_compound'] = sentiment_df['compound']
    
    # Compute additional features
    data_file['text_length'] = data_file['Text'].apply(len)
    if 'Summary' in data_file.columns:
        data_file['summary_length'] = data_file['Summary'].fillna('').apply(len)
    else:
        data_file['summary_length'] = 0

    if 'HelpfulnessNumerator' in data_file.columns and 'HelpfulnessDenominator' in data_file.columns:
        data_file['helpful_ratio'] = data_file.apply(
            lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] if row['HelpfulnessDenominator'] > 0 else 0,
            axis=1
        )
    else:
        data_file['helpful_ratio'] = 0
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Sentiment computation completed in {elapsed_time:.2f} seconds")
    
    # Save the updated DataFrame
    if output_path is None:
        output_path = data_path
    
    print(f"Saving updated dataset to {output_path}...")
    data_file.to_csv(output_path, index=False)
    print("Done!")
    
    # Print some statistics
    print("\nSentiment Score Statistics:")
    print(f"Negative (mean): {data_file['sentiment_neg'].mean():.4f}")
    print(f"Neutral (mean): {data_file['sentiment_neu'].mean():.4f}")
    print(f"Positive (mean): {data_file['sentiment_pos'].mean():.4f}")
    print(f"Compound (mean): {data_file['sentiment_compound'].mean():.4f}")
    
    # Count of positive, negative, and neutral reviews based on compound score
    positive_count = (data_file['sentiment_compound'] >= 0.05).sum()
    negative_count = (data_file['sentiment_compound'] <= -0.05).sum()
    neutral_count = ((data_file['sentiment_compound'] > -0.05) & (data_file['sentiment_compound'] < 0.05)).sum()
    
    print(f"\nVADER Sentiment Distribution:")
    print(f"Positive reviews: {positive_count} ({positive_count/len(data_file)*100:.2f}%)")
    print(f"Negative reviews: {negative_count} ({negative_count/len(data_file)*100:.2f}%)")
    print(f"Neutral reviews: {neutral_count} ({neutral_count/len(data_file)*100:.2f}%)")
    
    return data_file

# def main():
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Compute VADER sentiment for Amazon Food Reviews dataset')
#     parser.add_argument('--data_path', type=str, required=True, help='Path to the Amazon Food Reviews CSV file')
#     parser.add_argument('--output_path', type=str, help='Path to save the updated CSV file (optional)')
#     args = parser.parse_args()
    
#     # Compute sentiment scores
#     compute_vader_sentiment(args.data_path, args.output_path)

if __name__ == "__main__":
    data_path = "Data/cleaned_data.csv"
    output_path = "Data/cleaned_data_with_sentiment.csv"
    # main()
    compute_vader_sentiment(data_path, output_path)