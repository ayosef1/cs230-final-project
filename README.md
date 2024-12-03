# Sentiment Analysis of IMDb Movie Reviews

This project leverages Natural Language Processing (NLP) and Deep Learning to 
perform sentiment analysis on IMDb movie reviews. The objective is to 
classify reviews as either positive or negative based on their textual content.

---

## Project Overview

This project employs a deep learning approach using Long Short-Term Memory 
(LSTM) recurrent neural networks. Through extensive experimentation and 
hyperparameter tuning, the model is optimized to effectively understand and 
predict the sentiment expressed in movie reviews.

Key highlights:
- Built a robust preprocessing pipeline for text data.
- Incorporated word embeddings for enhanced feature representation.
- Fine-tuned hyperparameters using tools like Optuna.

---

## Dataset

The dataset contains **50,000 IMDb movie reviews**. We split the dataset into a
80/10/10 training/dev/testing split. The dataset is publicly available on Kaggle 
under open-access terms.

- **Dataset Link:** [IMDb Movie Review Dataset](https://www.kaggle.com/utathya/imdb-review-dataset)
- **Labels:** 
  - `Positive`: Reviews expressing favorable opinions.
  - `Negative`: Reviews expressing unfavorable opinions.

---

## Features

- **Preprocessing:** Tokenization, stop words removal, lemmatization, and Sequence padding for uniform input sizes
- **N-grams:** Unigrams and bigrams

---

## Models

### **LSTM Neural Network**

- Finetuned an LSTM-based deep learning model to capture sequential dependencies in text data.
- Incorporated dropout layers and early stopping to prevent overfitting.

---

## Results

Our best model achieved 89.54% accuracy on the test set.
