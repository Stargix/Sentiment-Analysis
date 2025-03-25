# Sentiment Analysis for Movie Reviews

This project implements and compares different approaches for sentiment analysis on movie reviews, using both supervised and unsupervised learning techniques. The goal is to automatically determine whether a movie review expresses a positive or negative opinion.

## Project Overview

This project is structured in two main parts:

1. **Supervised Learning Approach**: Using machine learning algorithms trained on labeled data to classify reviews
2. **Unsupervised Learning Approach**: Using SentiWordNet lexical resource without requiring labeled training data

## Dataset

The project uses the Movie Reviews Corpus from NLTK, which contains movie reviews labeled as either positive or negative. The dataset is preprocessed and split into training and testing sets with an 80/20 ratio.

## Methodology

### Text Preprocessing

Before training models, the text data undergoes several preprocessing steps:
- Conversion to lowercase
- Removal of newlines
- Removal of punctuation
- Tokenization
- Stopword removal
- Lemmatization

### Supervised Learning Approach

Several supervised machine learning models were implemented and evaluated:

- **Random Forest**: An ensemble of decision trees
- **K-Nearest Neighbors (KNN)**: Classification based on the k closest training examples
- **Support Vector Machine (SVM)**: Finding the optimal hyperplane to separate classes
- **Logistic Regression**: A linear model for binary classification
- **LightGBM**: A gradient boosting framework

Each model underwent hyperparameter optimization using Grid Search with 5-fold cross-validation to find the best configuration.

### Unsupervised Learning Approach

The unsupervised approach uses SentiWordNet, a lexical resource derived from WordNet that assigns sentiment scores to words:

- Each word is analyzed for its part of speech (POS) and assigned positive and negative scores
- Different aggregation methods were tested:
  - **Standard**: Sum all positive and negative scores directly
  - **Max**: Only sum the highest score (positive or negative) for each word
  - **Binary**: Count how many words are more positive than negative and vice versa
  - **Squared**: Square the scores to give more weight to intense sentiments
  - **Harmonic**: Use the harmonic mean between positive and negative scores

## Results

### Model Performance

- The supervised models significantly outperformed the unsupervised approach
- Random Forest achieved the best performance with 86.5% accuracy on the test set
- The unsupervised approach using SentiWordNet reached 65.2% accuracy
- Among different grammatical categories, adjectives proved most informative for sentiment detection

### Comparative Analysis

- Supervised models excel at capturing complex relationships between words and phrases
- Unsupervised approaches struggle with context, sarcasm, and mixed sentiments
- Even with small training sets, supervised models outperformed the lexicon-based approach
- An ROC curve analysis for the Random Forest model showed an AUC of 0.93, indicating excellent classification performance

## Installation & Requirements

```bash
pip install nltk scikit-learn pandas numpy matplotlib lightgbm
```

NLTK resources needed:
```python
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('movie_reviews')
```

## Usage

The project is implemented as a Jupyter notebook. To run it:

1. Install the required dependencies
2. Download the necessary NLTK resources
3. Run the notebook cells sequentially

## Error Analysis

Common classification errors include:
- Reviews with complex language and subtle nuances
- Reviews discussing negative aspects of a movie while being globally positive (or vice versa)
- Text containing sarcasm or irony
- Mixed sentiment reviews

## Authors

Sergi Flores & Sam Brumwell

## License

This project is for educational purposes and part of university coursework.
