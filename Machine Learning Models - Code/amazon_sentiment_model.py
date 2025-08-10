# Amazon Food Reviews Sentiment Analysis Model
# This script implements the machine learning pipeline for sentiment analysis on Amazon food reviews

# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import time
import warnings
import joblib
import argparse
from tqdm import tqdm

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Try importing optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Skipping XGBoost model.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Skipping LightGBM model.")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Skipping CatBoost model.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
for resource in ['stopwords', 'punkt', 'wordnet', 'vader_lexicon', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except:
        print(f"Warning: Could not download NLTK resource {resource}")

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

class AmazonSentimentModel:
    """Amazon Food Reviews Sentiment Analysis Model"""
    
    def __init__(self, data_path=None):
        """Initialize the model"""
        self.data_path = data_path
        self.preprocessor = None
        self.model = None
        self.pipeline = None
        self.sid = SentimentIntensityAnalyzer()
    
    def load_data(self, data_path=None):
        """Load the dataset"""
        if data_path is not None:
            self.data_path = data_path
        
        if self.data_path is None:
            raise ValueError("Data path not specified")
        
        print(f"Loading data from {self.data_path}...")
        self.data_file = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.data_file.shape}")
        
        return self.data_file
    
    def preprocess_data(self):
        """Preprocess the data"""
        print("Preprocessing data...")
        
        # Clean text data
        print("Cleaning text data...")
        self.data_file['cleaned_text'] = self.data_file['Text'].apply(lambda x: clean_text(x))
        self.data_file['cleaned_summary'] = self.data_file['Summary'].apply(lambda x: clean_text(x))
        
        # Create additional features
        print("Creating additional features...")
        
        # Text length features
        self.data_file['text_length'] = self.data_file['Text'].apply(lambda x: len(str(x)))
        self.data_file['summary_length'] = self.data_file['Summary'].apply(lambda x: len(str(x)))
        
        # Ratio of helpful votes
        self.data_file['helpful_ratio'] = self.data_file.apply(lambda x: 
                                                x['HelpfulnessNumerator'] / x['HelpfulnessDenominator'] 
                                                if x['HelpfulnessDenominator'] > 0 else 0, axis=1)
        
        # Sentiment features using NLTK's VADER - vectorized computation for the entire dataset
        print("Computing VADER sentiment for all reviews (this may take a while)...")
        
        # Ensure Text column has no NaN values
        self.data_file['Text'] = self.data_file['Text'].fillna('')
        
        # Use apply with lambda function to compute sentiment for each review
        sentiments = self.data_file['Text'].apply(lambda t: self.sid.polarity_scores(t))
        
        # Convert the series of dictionaries to a DataFrame
        sentiment_df = pd.DataFrame(sentiments.tolist(), index=self.data_file.index)
        
        # Add sentiment columns to the original DataFrame
        self.data_file['sentiment_neg'] = sentiment_df['neg']
        self.data_file['sentiment_neu'] = sentiment_df['neu']
        self.data_file['sentiment_pos'] = sentiment_df['pos']
        self.data_file['sentiment_compound'] = sentiment_df['compound']
        
        # Print some statistics
        print("\nSentiment Score Statistics:")
        print(f"Negative (mean): {self.data_file['sentiment_neg'].mean():.4f}")
        print(f"Neutral (mean): {self.data_file['sentiment_neu'].mean():.4f}")
        print(f"Positive (mean): {self.data_file['sentiment_pos'].mean():.4f}")
        print(f"Compound (mean): {self.data_file['sentiment_compound'].mean():.4f}")
        
        # Count of positive, negative, and neutral reviews based on compound score
        positive_count = (self.data_file['sentiment_compound'] >= 0.05).sum()
        negative_count = (self.data_file['sentiment_compound'] <= -0.05).sum()
        neutral_count = ((self.data_file['sentiment_compound'] > -0.05) & (self.data_file['sentiment_compound'] < 0.05)).sum()
        
        print(f"\nVADER Sentiment Distribution:")
        print(f"Positive reviews: {positive_count} ({positive_count/len(self.data_file)*100:.2f}%)")
        print(f"Negative reviews: {negative_count} ({negative_count/len(self.data_file)*100:.2f}%)")
        print(f"Neutral reviews: {neutral_count} ({neutral_count/len(self.data_file)*100:.2f}%)")

        
        # Note: We're using the existing Score column as our target variable
        # The Score column already has values 0 and 1
        self.data_file['target_binary'] = self.data_file['Score']
        self.data_file['target_multiclass'] = self.data_file['Score']
        
        return self.data_file
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("Preparing features for modeling...")
        
        # For simplicity, let's focus on just a subset of features
        # Numerical features
        self.num_features = ['HelpfulnessNumerator', 'HelpfulnessDenominator',
                       'text_length', 'summary_length', 'helpful_ratio']
        
        # Sentiment features
        self.sentiment_features = ['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']
        
        # All features for this simplified model
        self.all_features = self.num_features + self.sentiment_features
        
        # Split the data into training and testing sets
        self.X = self.data_file[self.all_features]
        self.y = self.data_file['Score']  # Using Score directly as target
        
        # For this script, we'll focus on binary classification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        print(f"Target distribution in training: {self.y_train.value_counts()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_preprocessor(self):
        """Create the feature preprocessing pipeline"""
        print("Creating preprocessing pipeline...")
        
        # Numerical features pipeline
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Combine all preprocessing steps using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.num_features + self.sentiment_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )
        
        # Fit the preprocessor on the training data
        print("Fitting preprocessor on training data...")
        self.X_train_preprocessed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_preprocessed = self.preprocessor.transform(self.X_test)
        
        print(f"Preprocessed training data shape: {self.X_train_preprocessed.shape}")
        print(f"Preprocessed testing data shape: {self.X_test_preprocessed.shape}")
        
        return self.preprocessor
    
    def train_models(self):
        """Train and evaluate multiple models"""
        print("Training models...")
        
        # Define models to evaluate with better hyperparameters
        self.models = {
            'Logistic Regression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        }
        
        # Add optional models if available with improved parameters
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2, random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42)
        
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = cb.CatBoostClassifier(n_estimators=200, depth=5, learning_rate=0.1, class_weights=[1, 2], random_state=42, verbose=0)
        
        # Evaluate each model
        self.results = []
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            result = self.evaluate_model(model, model_name)
            self.results.append(result)
        
        # Find the best model based on F1 score
        self.best_model_result = max(self.results, key=lambda x: x['f1'])
        self.best_model_name = self.best_model_result['model_name']
        self.best_model = self.best_model_result['model']
        
        print(f"\nBest model based on F1 score: {self.best_model_name}")
        print(f"F1 Score: {self.best_model_result['f1']:.4f}")
        print(f"Accuracy: {self.best_model_result['accuracy']:.4f}")
        
        return self.best_model, self.best_model_name
    
    def evaluate_model(self, model, model_name):
        """Evaluate a single model"""
        # Train the model
        start_time = time.time()
        model.fit(self.X_train_preprocessed, self.y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(self.X_test_preprocessed)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(self.X_test_preprocessed)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, digits=4))
        
        # Return metrics for comparison
        return {
            'model_name': model_name,
            'model': model,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'train_time': train_time,
            'probabilities': y_prob
        }
    
    def tune_hyperparameters(self):
        """Tune hyperparameters for the best model"""
        print(f"Performing hyperparameter tuning on {self.best_model_name}...")
        
        # Define hyperparameter grids for each model
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'num_leaves': [31, 50]
            },
            'CatBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'depth': [4, 6]
            }
        }
        
        # Get the parameter grid for the best model
        param_grid = param_grids.get(self.best_model_name, {})
        
        if not param_grid:
            print(f"No parameter grid defined for {self.best_model_name}. Skipping hyperparameter tuning.")
            self.best_model_tuned = self.best_model
            return self.best_model_tuned
        
        # Create a smaller subset of the training data for faster tuning
        X_train_sample, _, y_train_sample, _ = train_test_split(
            self.X_train_preprocessed, self.y_train, train_size=0.3, random_state=42, stratify=self.y_train)
        
        # Create and fit the grid search
        grid_search = GridSearchCV(
            estimator=self.best_model,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_sample, y_train_sample)
        
        # Get the best parameters and model
        best_params = grid_search.best_params_
        print(f"\nBest parameters: {best_params}")
        
        # Train the best model with the best parameters on the full training set
        self.best_model_tuned = grid_search.best_estimator_
        self.best_model_tuned.fit(self.X_train_preprocessed, self.y_train)
        
        # Evaluate the tuned model
        y_pred_tuned = self.best_model_tuned.predict(self.X_test_preprocessed)
        accuracy_tuned = accuracy_score(self.y_test, y_pred_tuned)
        f1_tuned = f1_score(self.y_test, y_pred_tuned)
        precision_tuned = precision_score(self.y_test, y_pred_tuned)
        recall_tuned = recall_score(self.y_test, y_pred_tuned)
        
        print(f"\nTuned {self.best_model_name} Results:")
        print(f"Accuracy: {accuracy_tuned:.4f}")
        print(f"F1 Score: {f1_tuned:.4f}")
        print(f"Precision: {precision_tuned:.4f}")
        print(f"Recall: {recall_tuned:.4f}")
        
        # Print confusion matrix
        cm_tuned = confusion_matrix(self.y_test, y_pred_tuned)
        print("\nConfusion Matrix:")
        print(cm_tuned)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_tuned))
        
        return self.best_model_tuned
    
    def save_model(self, model_path='amazon_reviews_sentiment_model.pkl'):
        """Save the final model"""
        print("Saving the final model...")
        
        # Create a complete pipeline including preprocessing and the model
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.best_model_tuned)
        ])
        
        # Fit the pipeline on the original training data
        self.pipeline.fit(self.X_train, self.y_train)
        
        # Save the pipeline
        joblib.dump(self.pipeline, model_path)
        print(f"Final model saved to {model_path}")
        
        # Test the saved model
        loaded_model = joblib.load(model_path)
        test_pred = loaded_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_pred)
        test_f1 = f1_score(self.y_test, test_pred)
        
        print(f"\nLoaded model test accuracy: {test_accuracy:.4f}")
        print(f"Loaded model test F1 score: {test_f1:.4f}")
        
        return model_path
    
    def predict_sentiment(self, review_text, review_summary=''):
        """Predict sentiment for a new review"""
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Create a DataFrame with the same structure as the training data
        data = pd.DataFrame({
            'ProductId': ['dummy_product'],
            'UserId': ['dummy_user'],
            'ProfileName': ['dummy_profile'],
            'HelpfulnessNumerator': [0],
            'HelpfulnessDenominator': [0],
            'Score': [0],  # Placeholder, not used for prediction
            'Time': [0],  # Placeholder, not used for prediction
            'Summary': [review_summary],
            'Text': [review_text],
            'usefulness': [0],  # Placeholder, not used for prediction
            'word_count': [len(review_text.split())],
            'cleaned_text': [clean_text(review_text)],
            'cleaned_summary': [clean_text(review_summary)],
            'text_length': [len(review_text)],
            'summary_length': [len(review_summary)],
            'helpful_ratio': [0],  # Placeholder, not used for prediction
            'sentiment_neg': [0],  # Placeholder, will be calculated
            'sentiment_neu': [0],  # Placeholder, will be calculated
            'sentiment_pos': [0],  # Placeholder, will be calculated
            'sentiment_compound': [0]  # Placeholder, will be calculated
        })
        
        # Calculate sentiment scores
        sentiment = self.sid.polarity_scores(review_text)
        data['sentiment_neg'] = sentiment['neg']
        data['sentiment_neu'] = sentiment['neu']
        data['sentiment_pos'] = sentiment['pos']
        data['sentiment_compound'] = sentiment['compound']
        
        # Make prediction
        prediction = self.pipeline.predict(data)[0]
        probabilities = self.pipeline.predict_proba(data)[0]
        
        # Determine confidence score - use the actual probability value
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Determine VADER sentiment for comparison
        vader_sentiment = "Positive" if sentiment['compound'] >= 0.05 else ("Negative" if sentiment['compound'] <= -0.05 else "Neutral")
        
        # Return results with enhanced information
        result = {
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'probability': confidence,
            'prediction': int(prediction),
            'sentiment_scores': sentiment,
            'vader_sentiment': vader_sentiment,
            'probabilities': probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
        }
        
        return result
    
    def run_pipeline(self, save_model=True, model_path='amazon_reviews_sentiment_model.pkl'):
        """Run the complete pipeline"""
        self.preprocess_data()
        self.prepare_features()
        self.create_preprocessor()
        self.train_models()
        self.tune_hyperparameters()
        
        if save_model:
            self.save_model(model_path)
        
        return self.pipeline

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Amazon Food Reviews Sentiment Analysis')
    parser.add_argument('--data_path', type=str, default='cleaned_data_with_sentiment.csv',
                        help='Path to the data file (default: cleaned_data_with_sentiment.csv)')
    parser.add_argument('--model_path', type=str, default='amazon_reviews_sentiment_model.pkl',
                        help='Path to save the model (default: amazon_reviews_sentiment_model.pkl)')
    parser.add_argument('--predict', action='store_true',
                        help='Run in prediction mode')
    parser.add_argument('--review', type=str,
                        help='Review text to predict sentiment for')
    parser.add_argument('--summary', type=str, default='',
                        help='Review summary to predict sentiment for')
    
    args = parser.parse_args()
    
    if args.predict:
        # Prediction mode
        if not args.review:
            print("Error: Review text is required for prediction mode.")
            return
        
        try:
            # Load the model
            print(f"Loading model from {args.model_path}...")
            pipeline = joblib.load(args.model_path)
            
            # Create a model instance
            model = AmazonSentimentModel()
            model.pipeline = pipeline
            
            # Make prediction
            result = model.predict_sentiment(args.review, args.summary)
            
            # Print results
            print("\nPrediction Results:")
            print(f"Review: {args.review}")
            if args.summary:
                print(f"Summary: {args.summary}")
            print(f"Predicted sentiment: {result['sentiment']}")
            print(f"Confidence: {result['probability']:.4f}")
            print(f"Sentiment scores: {result['sentiment_scores']}")
            
        except Exception as e:
            print(f"Error in prediction mode: {e}")
    else:
        # Training mode
        try:
            # Create and run the model
            model = AmazonSentimentModel(args.data_path)
            model.load_data()
            model.run_pipeline(save_model=True, model_path=args.model_path)
            
            # Test with some example reviews
            example_reviews = [
                {
                    'text': 'This product is amazing! I love it and would definitely recommend it to everyone.',
                    'summary': 'Great product!'
                },
                {
                    'text': 'Terrible experience. The product broke after one use and customer service was unhelpful.',
                    'summary': 'Disappointed'
                },
                {
                    'text': "It's okay. Not the best, not the worst. Might buy again if on sale.",
                    'summary': 'Average product'
                }
            ]
            
            print("\nExample predictions:")
            for i, review in enumerate(example_reviews):
                result = model.predict_sentiment(review['text'], review['summary'])
                print(f"\nExample {i+1}:")
                print(f"Text: {review['text']}")
                print(f"Summary: {review['summary']}")
                print(f"Predicted sentiment: {result['sentiment']}")
                print(f"Confidence: {result['probability']:.4f}")
                print(f"Sentiment scores: {result['sentiment_scores']}")
                
        except Exception as e:
            print(f"Error in training mode: {e}")

if __name__ == "__main__":
    main()