# Sentiment Analysis of IMDb Movie Reviews

This project applies Natural Language Processing (NLP) and Machine Learning (ML) techniques to perform sentiment analysis on a dataset of IMDb movie reviews. The goal is to classify reviews as either positive or negative based on their content.

## Project Overview

The project utilizes various machine learning models including Logistic Regression, Random Forest, XGBoost, and a deep learning approach using LSTM (Long Short-Term Memory) networks. Through a series of experiments, we explore the effectiveness of these models in understanding and predicting sentiments expressed in movie reviews.

## Dataset

The dataset consists of 50,000 IMDb movie reviews, split evenly into training and test sets. It is publicly available on Kaggle and was used under its open-access terms. 

[IMDb Movie Review Dataset](https://www.kaggle.com/utathya/imdb-review-dataset)

## Features

- **Preprocessing:** Tokenization, stop words removal, lemmatization, and TF-IDF vectorization.
- **N-grams:** Unigrams, bigrams, trigrams, and 4-grams.
- **Word Embeddings:** For the LSTM model, pretrained word embeddings were used.
- **Sequence Padding:** Applied to create uniform input sizes for LSTM.

## Models

- **Logistic Regression:** Served as a strong baseline with a comprehensive feature set.
- **Random Forest and XGBoost:** Utilized to explore tree-based ensemble methods.
- **LSTM Neural Network:** Implemented to capture sequential dependencies in text data.

## Results

Logistic Regression achieved the highest accuracy, indicating its efficiency in handling sparse data from TF-IDF vectorization. While LSTM showed promise, limited computational resources constrained extensive tuning, highlighting the importance of feature engineering and model simplicity.

## Installation

To run this project, clone the repo, and install the required dependencies:

```bash
git clone https://github.com/yourgithubusername/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
pip install -r requirements.txt
