#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amazon Food Reviews Sentiment Analysis Streamlit App

This script creates a web-based interface for the sentiment analysis model
using Streamlit, allowing users to analyze food review sentiments interactively.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page config
st.set_page_config(page_title="Amazon Food Reviews Sentiment Analyzer", page_icon="NNN", layout="wide")

# Ensure NLTK resources are downloaded
@st.cache_resource
def download_nltk_resources():
    for resource in ['stopwords', 'punkt', 'wordnet', 'vader_lexicon', 'omw-1.4']:
        try:
            nltk.download(resource, quiet=True)
        except:
            st.warning(f"Could not download NLTK resource {resource}")

# Text preprocessing functions
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

@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please train a model first.")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_sentiment(model, review_text, review_summary=''):
    """Predict sentiment for a new review using the loaded model"""
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
        st.error(f"Error making prediction: {e}")
        return None

def log_prediction(review, summary, result):
    """Log prediction for monitoring"""
    log_file = "prediction_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create log entry
    log_entry = pd.DataFrame({
        'timestamp': [timestamp],
        'review': [review],
        'summary': [summary],
        'prediction': [result['sentiment']],
        'confidence': [result['probability']],
        'vader_sentiment': [result['vader_sentiment']],
        'neg_score': [result['sentiment_scores']['neg']],
        'neu_score': [result['sentiment_scores']['neu']],
        'pos_score': [result['sentiment_scores']['pos']],
        'compound_score': [result['sentiment_scores']['compound']]
    })
    
    # Append to log file
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, log_entry], ignore_index=True)
    else:
        log_df = log_entry
    
    log_df.to_csv(log_file, index=False)

def show_monitoring_dashboard():
    """Display monitoring dashboard with prediction statistics"""
    log_file = "prediction_log.csv"
    
    if not os.path.exists(log_file):
        st.info("No prediction logs available yet. Make some predictions first.")
        return
    
    log_df = pd.read_csv(log_file)
    
    if len(log_df) == 0:
        st.info("No prediction logs available yet. Make some predictions first.")
        return
    
    st.subheader("Prediction Monitoring Dashboard")
    
    # Add timestamp column as datetime
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
    
    # Basic statistics
    st.write(f"Total predictions: {len(log_df)}")
    st.write(f"Last prediction: {log_df['timestamp'].max()}")
    
    # Prediction distribution
    st.subheader("Prediction Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model predictions
    prediction_counts = log_df['prediction'].value_counts()
    ax[0].pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax[0].set_title('Model Predictions')
    
    # VADER predictions
    vader_counts = log_df['vader_sentiment'].value_counts()
    ax[1].pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    ax[1].set_title('VADER Predictions')
    
    st.pyplot(fig)
    
    # Confidence distribution
    st.subheader("Confidence Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(log_df['confidence'], bins=20, kde=True, ax=ax)
    ax.set_title('Prediction Confidence Distribution')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    # Sentiment scores over time
    st.subheader("Sentiment Scores Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot sentiment scores
    ax.plot(log_df['timestamp'], log_df['compound_score'], label='Compound')
    ax.plot(log_df['timestamp'], log_df['pos_score'], label='Positive')
    ax.plot(log_df['timestamp'], log_df['neg_score'], label='Negative')
    
    ax.set_title('Sentiment Scores Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Score')
    ax.legend()
    st.pyplot(fig)
    
    # Recent predictions table
    st.subheader("Recent Predictions")
    recent_df = log_df.sort_values('timestamp', ascending=False).head(10)
    recent_df = recent_df[['timestamp', 'review', 'prediction', 'confidence', 'vader_sentiment']]
    st.dataframe(recent_df)

def main():
    # Download NLTK resources
    download_nltk_resources()
    
    # st.set_page_config(page_title="Amazon Food Reviews Sentiment Analyzer", page_icon="🍔", layout="wide")
    
    st.title("Amazon Food Reviews Sentiment Analyzer")
    st.markdown("""
    This app analyzes the sentiment of food reviews using a machine learning model trained on Amazon food reviews data.
    Enter your review text and optional summary to get a sentiment prediction.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Monitoring Dashboard", "About"])
    
    # Load model
    model_path = "amazon_reviews_sentiment_model.pkl"
    model = load_model(model_path)
    
    if page == "Sentiment Analysis":
        st.header("Analyze Review Sentiment")
        
        # Input form
        with st.form(key="sentiment_form"):
            review_text = st.text_area("Enter your review text:", height=150)
            review_summary = st.text_input("Enter a short summary (optional):")
            submit_button = st.form_submit_button(label="Analyze Sentiment")
        
        # Example reviews
        st.markdown("### Or try an example:")
        example_reviews = [
            {"text": "This product is amazing! I love it and would definitely recommend it to everyone.", "summary": "Great product!"},
            {"text": "Terrible experience. The product broke after one use and customer service was unhelpful.", "summary": "Disappointed"},
            {"text": "It's okay. Not the best, not the worst. Might buy again if on sale.", "summary": "Average product"}
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Positive Example"):
                review_text = example_reviews[0]["text"]
                review_summary = example_reviews[0]["summary"]
        with col2:
            if st.button("Negative Example"):
                review_text = example_reviews[1]["text"]
                review_summary = example_reviews[1]["summary"]
        with col3:
            if st.button("Neutral Example"):
                review_text = example_reviews[2]["text"]
                review_summary = example_reviews[2]["summary"]
        
        # Display the selected example
        if review_text:
            st.text_area("Review Text", review_text, height=100, key="display_review")
            if review_summary:
                st.text_input("Review Summary", review_summary, key="display_summary")
        
        # Process when submit button is clicked or example is selected
        if (submit_button or review_text) and model is not None:
            with st.spinner("Analyzing sentiment..."):
                # Add a small delay for better UX
                time.sleep(0.5)
                
                # Make prediction
                result = predict_sentiment(model, review_text, review_summary)
                
                if result:
                    # Log prediction
                    log_prediction(review_text, review_summary, result)
                    
                    # Display results
                    st.subheader("Sentiment Analysis Results")
                    
                    # Create columns for main results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display sentiment with color
                        sentiment_color = "#1E8449" if result['sentiment'] == "Positive" else "#C0392B"
                        st.markdown(f"<h3 style='color: {sentiment_color};'>Prediction: {result['sentiment']}</h3>", unsafe_allow_html=True)
                        
                        # Display confidence
                        confidence_percentage = result['probability'] * 100
                        st.markdown(f"<h4>Confidence: {confidence_percentage:.1f}%</h4>", unsafe_allow_html=True)
                        
                        # Progress bar for confidence
                        st.progress(result['probability'])
                    
                    with col2:
                        # Display VADER sentiment
                        vader_color = "#1E8449" if result['vader_sentiment'] == "Positive" else ("#C0392B" if result['vader_sentiment'] == "Negative" else "#7D3C98")
                        st.markdown(f"<h3 style='color: {vader_color};'>VADER: {result['vader_sentiment']}</h3>", unsafe_allow_html=True)
                        
                        # Note if model and VADER disagree
                        if result['sentiment'] != result['vader_sentiment'] and result['vader_sentiment'] != 'Neutral':
                            st.warning("The ML model and VADER lexicon disagree on the sentiment. This might indicate a complex or nuanced review.")
                    
                    # Display detailed sentiment scores
                    st.subheader("Detailed Sentiment Scores")
                    
                    # Create a DataFrame for the scores
                    scores_df = pd.DataFrame({
                        'Score Type': ['Negative', 'Neutral', 'Positive', 'Compound'],
                        'Value': [
                            result['sentiment_scores']['neg'],
                            result['sentiment_scores']['neu'],
                            result['sentiment_scores']['pos'],
                            result['sentiment_scores']['compound']
                        ]
                    })
                    
                    # Plot the scores
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(
                        scores_df['Score Type'],
                        scores_df['Value'],
                        color=['#C0392B', '#7D3C98', '#1E8449', '#2E86C1']
                    )
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom')
                    
                    ax.set_ylim(0, 1.1)  # Set y-axis limit
                    ax.set_title('VADER Sentiment Scores')
                    ax.set_ylabel('Score')
                    st.pyplot(fig)
                    
                    # Explanation of scores
                    with st.expander("What do these scores mean?"):
                        st.markdown("""
                        - **Negative**: Proportion of negative sentiment in the text (0 to 1)
                        - **Neutral**: Proportion of neutral sentiment in the text (0 to 1)
                        - **Positive**: Proportion of positive sentiment in the text (0 to 1)
                        - **Compound**: Normalized score that summarizes all the sentiment scores (-1 to 1)
                            - -1 is extremely negative
                            - 0 is neutral
                            - +1 is extremely positive
                        """)
    
    elif page == "Monitoring Dashboard":
        st.header("Monitoring Dashboard")
        show_monitoring_dashboard()
    
    elif page == "About":
        st.header("About This App")
        st.markdown("""
        ### Amazon Food Reviews Sentiment Analyzer
        
        This application uses a machine learning model trained on Amazon food reviews to predict the sentiment of new reviews.
        
        #### Features:
        - **Sentiment Analysis**: Predicts whether a review is positive or negative
        - **Confidence Score**: Provides a confidence level for the prediction
        - **VADER Sentiment**: Compares the ML model prediction with a lexicon-based approach
        - **Detailed Scores**: Breaks down the sentiment components
        - **Monitoring**: Tracks predictions and model performance over time
        
        #### Model Information:
        - Trained on Amazon food reviews dataset
        - Uses both numerical features and sentiment scores
        - Implements a pipeline with preprocessing and classification
        
        #### Monitoring and Concept Drift:
        The app logs all predictions to track model performance over time. This helps identify potential concept drift, where the model's performance degrades as language patterns change. Regular retraining on new data is recommended to maintain accuracy.
        
        #### Created By:
        This application was developed as part of a machine learning project for sentiment analysis.
        #### By: 
        #####     Norden Sherpa
        """)

if __name__ == "__main__":
    main()