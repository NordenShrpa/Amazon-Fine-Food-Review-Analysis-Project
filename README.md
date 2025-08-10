# Amazon Food Reviews Sentiment Analysis Project

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

## Overview

This project provides a comprehensive toolkit for sentiment analysis on the Amazon Fine Food Reviews dataset. It encompasses exploratory data analysis (EDA), data preprocessing, VADER-based sentiment computation, supervised machine learning modeling, unsupervised clustering, model interpretability with SHAP, and an interactive Streamlit web application for real-time predictions. The goal is to classify reviews as positive or negative while offering insights into customer sentiments.

Key features:
- **Data Processing**: Cleaning, feature engineering, and VADER sentiment scoring.
- **Modeling**: Binary classification using logistic regression, random forests, and optional boosted models (XGBoost, LightGBM, CatBoost).
- **Analysis**: Unsupervised clustering with KMeans and explainability via SHAP.
- **Deployment**: CLI tools for batch processing and a user-friendly Streamlit app with monitoring dashboard.
- **Monitoring**: Prediction logging to detect concept drift over time.

The project follows modular design principles, with scripts organized for easy extension and maintenance.

## Dataset

The dataset used is the [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) from Kaggle (not included in this repo due to size). It contains ~568,000 reviews with columns like `Text`, `Summary`, `Score`, and helpfulness metrics. Processed versions (e.g., `cleaned_data.csv`) are generated during execution and stored in the Data folder.

## Installation

1. Clone the repository:
git clone https://github.com/NordenShrpa/Amazon-Fine-Food-Review-Analysis-Project.git
cd amazon-food-reviews-sentiment-analysis
text2. Create and activate a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
text3. Install dependencies:
pip install -r requirements.txt
textNote: For optional models (XGBoost, LightGBM, CatBoost), install them separately if needed: `pip install xgboost lightgbm catboost`.

4. Download NLTK resources (run once):
python -c "import nltk; nltk.download(['stopwords', 'punkt', 'wordnet', 'vader_lexicon', 'omw-1.4'])"
text## Usage

### 1. Exploratory Data Analysis
Run the Jupyter notebook for insights:
jupyter notebook EDA_amazon_food_review.ipynb
text### 2. Train the Model
Train and save the sentiment model:
python amazon_sentiment_model.py --data_path Data/cleaned_data_with_sentiment.csv
textThis generates `amazon_reviews_sentiment_model.pkl` in the root.

### 3. Compute VADER Sentiments (Batch Mode)
Add sentiment scores to a dataset:
python compute_vader_sentiment.py --data_path Data/cleaned_data.csv --output_path Data/cleaned_data_with_sentiment.csv
text### 4. Analyze Single Review (CLI Mode)
Predict sentiment for a review (using the combined toolkit script):
python sentiment_analysis.py analyze --review "This product is amazing!" --summary "Great buy" --model amazon_reviews_sentiment_model.pkl
textInteractive mode:
python sentiment_analysis.py analyze --interactive --model amazon_reviews_sentiment_model.pkl
textFor legacy single-review analysis (using the standalone script):
python analyze_sentiment.py --review "This product is amazing!" --summary "Great buy" --model amazon_reviews_sentiment_model.pkl
text### 5. Run the Streamlit App
Launch the web interface:
streamlit run app.py
textAccess at http://localhost:8501. Features include real-time prediction, monitoring dashboard, and about section.

### 6. Unsupervised Clustering
Perform KMeans analysis:
python unsupervised_analysis.py --data_path Data/cleaned_data.csv
textOutputs include `clustered_data.csv`, `cluster_analysis.csv`, `optimal_clusters.png`, and `clusters_visualization.png`.

### 7. SHAP Model Explanation
Generate global/local explanations:
python shap_analysis.py --data_path Data/cleaned_data.csv --model_path amazon_reviews_sentiment_model.pkl
textOutputs include `shap_global_importance.png` and local explanation PNGs like `shap_local_explanation_0.png`.

### 8. Organize Project Structure
Restructure files into a more hierarchical format (optional; run with dry-run first to preview changes):
python organize_project.py --dry-run
textThis moves files into subfolders like notebooks/, scripts/, src/, data/processed/, and models/.

## Project Structure

The project uses a mostly flat structure in the root directory, with generated data files in a dedicated `Data` folder and potential output artifacts (like PNG visualizations or CSVs) also in root unless reorganized. Below is a detailed, discrete listing of key files and folders, including their purposes and any notes on generation or usage:

- **amazon_sentiment_model.py**: Core script for training the machine learning model. Handles data loading, preprocessing, feature preparation, model training (including hyperparameter tuning), and saving the pipeline. Supports optional boosted models if installed.
  
- **analyze_sentiment.py**: Standalone CLI script for analyzing sentiment of individual reviews. Provides interactive mode and single-review predictions using the trained model and VADER for comparison.

- **app.py**: Streamlit web application for interactive sentiment analysis. Includes pages for prediction, monitoring dashboard (with logged predictions), and project information. Logs predictions to `prediction_log.csv` for concept drift monitoring.

- **compute_vader_sentiment.py**: Script to compute VADER sentiment scores in batch mode for the entire dataset. Adds sentiment features to CSVs and saves updated files in the Data folder.

- **EDA_amazon_food_review.ipynb**: Jupyter notebook for exploratory data analysis. Covers data loading, cleaning, visualizations (e.g., score distributions), and initial insights. Outputs cleaned data to `Data/cleaned_data.csv`.

- **organize_project.py**: Utility script to reorganize the project into a more structured hierarchy (e.g., notebooks/, scripts/, src/). Supports dry-run mode for previewing changes without modifying files.

- **sentiment_analysis.py**: Combined toolkit script for sentiment tasks. Supports both batch VADER computation and individual review analysis via CLI, integrating functionalities from other scripts for efficiency.

- **shap_analysis.py**: Script for model interpretability using SHAP. Generates global feature importance plots and local explanations for samples, saving PNG outputs like `shap_global_importance.png`.

- **unsupervised_analysis.py**: Script for unsupervised learning, focusing on KMeans clustering of reviews based on features. Determines optimal clusters, visualizes results, and saves clustered data and analysis CSVs.

- **Data/** (folder): Contains processed datasets and generated files.
  - `cleaned_data.csv`: Preprocessed dataset from EDA, with cleaned text and basic features.
  - `cleaned_data_with_sentiment.csv`: Dataset enhanced with VADER scores from compute_vader_sentiment.py.
  - Other potential files: May include additional CSVs like `clustered_data.csv` generated during runs.

- **Generated Artifacts** (in root, unless reorganized):
  - `amazon_reviews_sentiment_model.pkl`: Saved trained model pipeline from amazon_sentiment_model.py.
  - `cluster_analysis.csv`: Summary statistics from clustering in unsupervised_analysis.py.
  - `clustered_data.csv`: Dataset with cluster labels from unsupervised_analysis.py.
  - `clusters_visualization.png`: Visualization of clustering results (PCA scatter, distributions, etc.).
  - `optimal_clusters.png`: Elbow and silhouette plots for optimal cluster determination.
  - `prediction_log.csv`: Log of predictions from the Streamlit app for monitoring.
  - `shap_global_importance.png`: Global SHAP feature importance plot.
  - `shap_local_explanation_*.png`: Local SHAP waterfall plots for specific samples (e.g., shap_local_explanation_0.png).
  - Other: May include catboost_info/ folder if CatBoost is used during training.

- **requirements.txt**: List of project dependencies with pinned versions for reproducibility.

- **README.md**: This documentation file, providing overview, installation, usage, and structure details.

- **.gitignore** (if present): Specifies files/folders to ignore in Git, such as large CSVs, models, and virtual environments.

Note: Running organize_project.py will move files into subfolders (e.g., notebooks/, scripts/, src/, data/processed/, models/, tests/). Update this section after reorganization if needed. Generated files like PNGs and CSVs are not version-controlled due to size.

## Monitoring and Concept Drift

Predictions are logged to `prediction_log.csv` in the Streamlit app. Monitor for drift by analyzing shifts in sentiment distribution over time. Retrain periodically with new data.

## Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/new-feature`.
3. Commit changes: `git commit -am 'Add new feature'`.
4. Push: `git push origin feature/new-feature`.
5. Submit a Pull Request.

Please follow PEP8 style guidelines and add tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: Snap Amazon Fine Food Reviews on Kaggle.
- Libraries: scikit-learn, NLTK, SHAP, Streamlit.
- Inspired by standard ML workflows for NLP tasks.
