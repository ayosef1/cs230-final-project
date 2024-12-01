import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lstm_nn import LSTMClassifier

def main():
    raw_data = 'data/imdb_dataset.csv'
    processed_data_file = 'data/imdb_dataset.pkl'

    model = LSTMClassifier(raw_data, processed_data_file)

    model.train()
    model.test()
    
if __name__ == "__main__":
    main()