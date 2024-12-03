import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lstm_nn import LSTMClassifier

def main():
    raw_data = 'data/imdb_dataset.csv'
    processed_data_file = 'data/imdb_dataset.pkl'

    embedding_dim = 100
    lstm_units = 64
    dropout_rate = 0.4
    dense_units = 96
    batch_size = 128
    learning_rate = 0.0001681035160630061

    model = LSTMClassifier(
            raw_data=raw_data,
            processed_data_file=processed_data_file,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            batch_size=batch_size,
            epochs=10,
            learning_rate=learning_rate,
            checkpoint='trial14.h5'
        )
    
    model.train()
    print("TRAINING COMPLETE\n\n\n")
    model.test()
    
if __name__ == "__main__":
    main()