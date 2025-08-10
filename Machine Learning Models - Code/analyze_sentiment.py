#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amazon Food Reviews Sentiment Analysis CLI

This script provides a simple command-line interface for analyzing the sentiment of food reviews.
It uses the trained model from amazon_sentiment_model.py to make predictions.
"""

import argparse
import joblib
import sys
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
for resource in ['stopwords', 'punkt', 'wordnet', 'vader_lexicon', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except:
        print(f"Warning: Could not download NLTK resource {resource}")

# Text preprocessing functions (same as in the main model)
def decontracted(phrase):
    """Expand contractions in text"""
    # Specific contractions
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "cannot", phrase)
    phrase = re.sub(r"n't", " not", phrase)
    phrase = re.sub(r"'re", " are", phrase)
    phrase = re.sub(r"'s", " is", phrase)
    phrase = re.sub(r"'d", " would", phrase)
    phrase = re.sub(r"'ll", " will", phrase)
    phrase = re.sub(r"'t", " not", phrase)
    phrase = re.sub(r"'ve", " have", phrase)
    phrase = re.sub(r"'m", " am", phrase)
    return phrase

def clean_text(text):
    """Clean and normalize text"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        text = decontracted(text)
        
        # Remove HTML tags
        text = re.sub('<.*?>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove words with digits
        text = re.sub('\w*\d\w*', ' ', text)
        
        # Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        
        # Remove extra whitespace
        text = re.sub('\s+', ' ', text).strip()
        
        # Simple tokenization without using word_tokenize
        tokens = text.split()
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except:
            # If stopwords fail, continue without removing them
            pass
        
        # Lemmatize
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except:
            # If lemmatization fails, continue without it
            pass
        
        # Join tokens back into text
        text = ' '.join(tokens)
        
        return text
    else:
        return ''

def predict_sentiment(model, review_text, review_summary=''):
    """Predict sentiment for a new review using the loaded model"""
    import pandas as pd
    
    # Create a sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Calculate sentiment scores
    sentiment = sid.polarity_scores(review_text)
    
    # Create a DataFrame with only the features used in the model
    data = pd.DataFrame({
        'HelpfulnessNumerator': [0],
        'HelpfulnessDenominator': [0],
        'text_length': [len(review_text)],
        'summary_length': [len(review_summary)],
        'helpful_ratio': [0],  # Placeholder, not used for prediction
        'sentiment_neg': [sentiment['neg']],
        'sentiment_neu': [sentiment['neu']],
        'sentiment_pos': [sentiment['pos']],
        'sentiment_compound': [sentiment['compound']]
    })
    
    # Make prediction
    try:
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        
        # Determine sentiment label based on prediction
        sentiment_label = "Positive" if prediction == 1 else "Negative"
        
        # Determine VADER sentiment for comparison
        vader_sentiment = "Positive" if sentiment['compound'] >= 0.05 else ("Negative" if sentiment['compound'] <= -0.05 else "Neutral")
        
        # Return results
        result = {
            'sentiment': sentiment_label,
            'probability': probabilities[1] if prediction == 1 else probabilities[0],
            'prediction': int(prediction),
            'sentiment_scores': sentiment,
            'vader_sentiment': vader_sentiment,
            'probabilities': probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
        }
        
        return result
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def interactive_mode(model):
    """Run the sentiment analyzer in interactive mode"""
    print("\n===== Amazon Food Review Sentiment Analyzer =====")
    print("Type 'exit' to quit at any time.")
    
    while True:
        print("\n")
        review = input("Enter your review text: ")
        
        if review.lower() == 'exit':
            print("Exiting interactive mode.")
            break
        
        summary = input("Enter review summary (optional): ")
        
        if summary.lower() == 'exit':
            print("Exiting interactive mode.")
            break
        
        result = predict_sentiment(model, review, summary)
        
        if result:
            print("\nSentiment Analysis Results:")
            print(f"Review: {review}")
            if summary:
                print(f"Summary: {summary}")
            print(f"\nModel Prediction: {result['sentiment']}")
            print(f"Confidence: {result['probability']:.4f} ({result['probability']*100:.1f}%)")
            print(f"\nVADER Sentiment: {result['vader_sentiment']}")
            print("\nDetailed sentiment scores:")
            print(f"  Negative: {result['sentiment_scores']['neg']:.3f}")
            print(f"  Neutral: {result['sentiment_scores']['neu']:.3f}")
            print(f"  Positive: {result['sentiment_scores']['pos']:.3f}")
            print(f"  Compound: {result['sentiment_scores']['compound']:.3f}")
            
            # If model and VADER disagree, note this
            if result['sentiment'] != result['vader_sentiment'] and result['vader_sentiment'] != 'Neutral':
                print("\nNote: The ML model and VADER lexicon disagree on the sentiment.")
                print("This might indicate a complex or nuanced review.")

def main():
    parser = argparse.ArgumentParser(description='Amazon Food Reviews Sentiment Analysis CLI')
    parser.add_argument('--model', type=str, default='amazon_reviews_sentiment_model.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--review', type=str,
                        help='Review text to analyze')
    parser.add_argument('--summary', type=str, default='',
                        help='Review summary (optional)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Please train a model first using amazon_sentiment_model.py")
        sys.exit(1)
    
    # Load the model
    try:
        print(f"Loading model from {args.model}...")
        model = joblib.load(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Interactive mode or single prediction
    if args.interactive:
        interactive_mode(model)
    elif args.review:
        result = predict_sentiment(model, args.review, args.summary)
        
        if result:
            print("\nSentiment Analysis Results:")
            print(f"Review: {args.review}")
            if args.summary:
                print(f"Summary: {args.summary}")
            print(f"\nModel Prediction: {result['sentiment']}")
            print(f"Confidence: {result['probability']:.4f} ({result['probability']*100:.1f}%)")
            print(f"\nVADER Sentiment: {result['vader_sentiment']}")
            print("\nDetailed sentiment scores:")
            print(f"  Negative: {result['sentiment_scores']['neg']:.3f}")
            print(f"  Neutral: {result['sentiment_scores']['neu']:.3f}")
            print(f"  Positive: {result['sentiment_scores']['pos']:.3f}")
            print(f"  Compound: {result['sentiment_scores']['compound']:.3f}")
            
            # If model and VADER disagree, note this
            if result['sentiment'] != result['vader_sentiment'] and result['vader_sentiment'] != 'Neutral':
                print("\nNote: The ML model and VADER lexicon disagree on the sentiment.")
                print("This might indicate a complex or nuanced review.")
    else:
        print("Error: Please provide a review text or use interactive mode.")
        print("Use --review \"Your review text\" or --interactive")

if __name__ == "__main__":
    main()