#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Organization Script

This script organizes the Amazon Food Reviews Sentiment Analysis project
into a cleaner, more maintainable structure following best practices.
"""

import os
import shutil
import argparse

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def organize_project(base_dir, dry_run=False):
    """Organize project files into a cleaner structure"""
    # Define the directory structure
    directories = [
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "notebooks"),
        os.path.join(base_dir, "src", "data"),
        os.path.join(base_dir, "src", "features"),
        os.path.join(base_dir, "src", "models"),
        os.path.join(base_dir, "src", "visualization"),
        os.path.join(base_dir, "src", "analysis"),
        os.path.join(base_dir, "src", "app"),
        os.path.join(base_dir, "scripts"),
        os.path.join(base_dir, "tests"),
    ]
    
    # Create directories
    if not dry_run:
        for directory in directories:
            create_directory(directory)
    else:
        print("\nDirectories that would be created:")
        for directory in directories:
            print(f"  {directory}")
    
    # Define file movements
    file_movements = [
        # Data files
        ("cleaned_data.csv", os.path.join("data", "processed", "cleaned_data.csv")),
        ("cleaned_data_with_sentiment.csv", os.path.join("data", "processed", "cleaned_data_with_sentiment.csv")),
        
        # Model files
        ("amazon_reviews_sentiment_model.pkl", os.path.join("models", "amazon_reviews_sentiment_model.pkl")),
        
        # Notebooks
        ("EDA_amazon_food_review.ipynb", os.path.join("notebooks", "01_exploratory_data_analysis.ipynb")),
        ("model_amazon.ipynb", os.path.join("notebooks", "02_model_development.ipynb")),
        
        # Scripts
        ("sentiment_analysis.py", os.path.join("scripts", "sentiment_analysis.py")),
        ("amazon_sentiment_model.py", os.path.join("scripts", "train_model.py")),
        
        # Source code
        ("app.py", os.path.join("src", "app", "streamlit_app.py")),
        ("shap_analysis.py", os.path.join("src", "analysis", "shap_explainer.py")),
        ("unsupervised_analysis.py", os.path.join("src", "analysis", "clustering.py")),
    ]
    
    # Move files
    if not dry_run:
        print("\nMoving files:")
        for source, destination in file_movements:
            source_path = os.path.join(base_dir, source)
            destination_path = os.path.join(base_dir, destination)
            
            if os.path.exists(source_path):
                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(source_path, destination_path)
                print(f"  Copied: {source} -> {destination}")
            else:
                print(f"  Warning: Source file not found: {source}")
    else:
        print("\nFiles that would be moved:")
        for source, destination in file_movements:
            print(f"  {source} -> {destination}")
    
    # Create __init__.py files
    init_files = [
        os.path.join("src", "__init__.py"),
        os.path.join("src", "data", "__init__.py"),
        os.path.join("src", "features", "__init__.py"),
        os.path.join("src", "models", "__init__.py"),
        os.path.join("src", "visualization", "__init__.py"),
        os.path.join("src", "analysis", "__init__.py"),
        os.path.join("src", "app", "__init__.py"),
        os.path.join("tests", "__init__.py"),
    ]
    
    if not dry_run:
        print("\nCreating __init__.py files:")
        for init_file in init_files:
            init_path = os.path.join(base_dir, init_file)
            with open(init_path, 'w') as f:
                f.write("")
            print(f"  Created: {init_file}")
    else:
        print("\n__init__.py files that would be created:")
        for init_file in init_files:
            print(f"  {init_file}")
    
    # Create module files
    module_files = [
        (os.path.join("src", "data", "preprocessing.py"), """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Preprocessing Module

This module contains functions for cleaning and preprocessing text data.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    """Download required NLTK resources"""
    for resource in ['stopwords', 'punkt', 'wordnet', 'vader_lexicon', 'omw-1.4']:
        try:
            nltk.download(resource, quiet=True)
        except:
            print(f"Could not download NLTK resource {resource}")

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
"""),
        
        (os.path.join("src", "features", "sentiment.py"), """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analysis Module

This module contains functions for sentiment analysis using VADER.
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import time

# Ensure VADER lexicon is downloaded
def ensure_vader_lexicon():
    """Ensure VADER lexicon is downloaded"""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

def compute_vader_sentiment(data_file, text_column='Text'):
    """Compute VADER sentiment scores for all reviews in the dataset"""
    # Ensure VADER lexicon is available
    ensure_vader_lexicon()
    
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Start timer
    start_time = time.time()
    
    # Create a copy of the dataframe to avoid modifying the original
    df = data_file.copy()
    
    # Handle NaN values in the text column
    df[text_column] = df[text_column].fillna('')
    
    # Compute sentiment scores for all reviews
    print(f"Computing VADER sentiment for {len(df)} reviews...")
    
    # Use list comprehension with tqdm for progress tracking
    sentiment_scores = [sid.polarity_scores(text) for text in tqdm(df[text_column], desc="Processing")]
    
    # Add sentiment scores to the dataframe
    df['sentiment_neg'] = [score['neg'] for score in sentiment_scores]
    df['sentiment_neu'] = [score['neu'] for score in sentiment_scores]
    df['sentiment_pos'] = [score['pos'] for score in sentiment_scores]
    df['sentiment_compound'] = [score['compound'] for score in sentiment_scores]
    
    # End timer
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Sentiment computation completed in {processing_time:.2f} seconds.")
    
    # Print sentiment statistics
    print("\nSentiment Statistics:")
    print(f"Mean negative score: {df['sentiment_neg'].mean():.4f}")
    print(f"Mean neutral score: {df['sentiment_neu'].mean():.4f}")
    print(f"Mean positive score: {df['sentiment_pos'].mean():.4f}")
    print(f"Mean compound score: {df['sentiment_compound'].mean():.4f}")
    
    # Determine sentiment distribution
    df['vader_sentiment'] = df['sentiment_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    sentiment_counts = df['vader_sentiment'].value_counts()
    total = len(df)
    
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        print(f"{sentiment.capitalize()} reviews: {count} ({percentage:.2f}%)")
    
    return df
"""),
    ]
    
    if not dry_run:
        print("\nCreating module files:")
        for module_file, content in module_files:
            module_path = os.path.join(base_dir, module_file)
            with open(module_path, 'w') as f:
                f.write(content.strip())
            print(f"  Created: {module_file}")
    else:
        print("\nModule files that would be created:")
        for module_file, _ in module_files:
            print(f"  {module_file}")
    
    # Create data README
    data_readme = os.path.join(base_dir, "data", "README.md")
    data_readme_content = """
# Data Directory

This directory contains the datasets used in the Amazon Food Reviews Sentiment Analysis project.

## Contents

### Processed Data

- `cleaned_data.csv`: Preprocessed Amazon food reviews dataset
- `cleaned_data_with_sentiment.csv`: Preprocessed dataset with VADER sentiment scores

## Data Description

The dataset contains Amazon food reviews with the following columns:

- `Id`: Unique review identifier
- `ProductId`: Unique product identifier
- `UserId`: Unique user identifier
- `ProfileName`: User profile name
- `HelpfulnessNumerator`: Number of users who found the review helpful
- `HelpfulnessDenominator`: Number of users who indicated whether the review was helpful
- `Score`: Rating (1-5)
- `Time`: Review timestamp
- `Summary`: Review summary
- `Text`: Full review text
- `text_length`: Length of the review text
- `summary_length`: Length of the review summary
- `helpful_ratio`: Ratio of HelpfulnessNumerator to HelpfulnessDenominator
- `sentiment_neg`: VADER negative sentiment score
- `sentiment_neu`: VADER neutral sentiment score
- `sentiment_pos`: VADER positive sentiment score
- `sentiment_compound`: VADER compound sentiment score
- `target_binary`: Binary sentiment (1 for positive, 0 for negative)
- `target_multiclass`: Multi-class sentiment (1-5 corresponding to Score)
"""
    
    if not dry_run:
        with open(data_readme, 'w') as f:
            f.write(data_readme_content.strip())
        print(f"\nCreated data README: {data_readme}")
    else:
        print(f"\nData README would be created: {data_readme}")
    
    # Create setup.py
    setup_py = os.path.join(base_dir, "setup.py")
    setup_py_content = """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="amazon_sentiment_analysis",
    version="0.1.0",
    description="Sentiment Analysis for Amazon Food Reviews",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "matplotlib",
        "seaborn",
        "joblib",
        "tqdm",
        "streamlit",
        "shap",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "optional": [
            "xgboost",
            "lightgbm",
            "catboost",
        ],
    },
    python_requires=">=3.6",
)
"""
    
    if not dry_run:
        with open(setup_py, 'w') as f:
            f.write(setup_py_content.strip())
        print(f"\nCreated setup.py: {setup_py}")
    else:
        print(f"\nsetup.py would be created: {setup_py}")
    
    # Create .gitignore
    gitignore = os.path.join(base_dir, ".gitignore")
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
.env/

# Jupyter Notebook
.ipynb_checkpoints

# IDE files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db

# Large data files
*.csv
*.pkl
*.h5

# Logs
logs/
*.log

# Catboost info
catboost_info/
"""
    
    if not dry_run:
        with open(gitignore, 'w') as f:
            f.write(gitignore_content.strip())
        print(f"\nCreated .gitignore: {gitignore}")
    else:
        print(f"\n.gitignore would be created: {gitignore}")
    
    print("\nProject organization complete!")
    if dry_run:
        print("\nThis was a dry run. No files were actually moved or created.")
        print("Run without --dry-run to actually reorganize the project.")

def main():
    parser = argparse.ArgumentParser(description='Organize Amazon Food Reviews Sentiment Analysis project')
    parser.add_argument('--base_dir', type=str, default=os.getcwd(),
                        help='Base directory of the project (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform a dry run without making any changes')
    
    args = parser.parse_args()
    
    print(f"Organizing project in: {args.base_dir}")
    organize_project(args.base_dir, args.dry_run)

if __name__ == "__main__":
    main()