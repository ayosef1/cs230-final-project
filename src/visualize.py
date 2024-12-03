import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
from src.lstm_nn import LSTMClassifier
from preprocessor import Preprocessor

def evaluate_and_visualize(models, X_test, y_test):
    """
    Evaluate models, plot confusion matrices, and ROC curves.

    Parameters:
        models (list): List of paths to the `.h5` model files.
        X_test (np.ndarray): Test dataset features.
        y_test (np.ndarray): Test dataset labels (0 or 1).

    Returns:
        None
    """
    for model_file in models:
        # Load the model
        model = load_model(model_file)
        print(f"Evaluating Model: {model_file}")
        
        # Predict probabilities and binary labels
        y_pred_proba = model.predict(X_test).flatten()
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred_binary, model_file)
        
        # ROC Curve
        plot_roc_curve(y_test, y_pred_proba, model_file)

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots a confusion matrix as a heatmap.

    Parameters:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted binary labels.
        model_name (str): Name or path of the model.

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f"{model_name}_confusion_matrix.png")  # Save the plot
    plt.close()  # Close the plot to free memory

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """
    Plots the ROC curve.

    Parameters:
        y_true (np.ndarray): Ground truth labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
        model_name (str): Name or path of the model.

    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Random guessing line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(f"{model_name}_roc_curve.png")  # Save the plot
    plt.close()  # Close the plot to free memory

# Example Usage
if __name__ == "__main__":
    # Paths to your .h5 model files
    model_files = ["trial10.h5", "trial14.h5", "trial5.h5"]
    raw_data = 'data/imdb_dataset.csv'
    processed_data_file = 'data/imdb_dataset.pkl'

    df = Preprocessor.preprocess(raw_data, processed_data_file)
    
    model = LSTMClassifier(raw_data=raw_data, processed_data_file=processed_data_file)
    X_train_vect, y_train, X_dev_vect, y_dev, X_test, y_test, vocab_size_train = model._LSTMClassifier__preprocess_for_lstm(df)

    # Evaluate and visualize
    evaluate_and_visualize(model_files, X_test, y_test)
