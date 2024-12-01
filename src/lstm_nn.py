import numpy as np
from src.preprocessor import Preprocessor
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class LSTMClassifier:
    def __init__(self,
                 raw_data, 
                 processed_data_file, 
                 embedding_dim=50, 
                 lstm_units=32, 
                 dropout_rate=0.5,
                 dense_units=32,
                 batch_size=32, 
                 epochs=1,
                 learning_rate=0.01,
                 X_test_vect=None,
                 y_test=None):
        self.raw_data = raw_data
        self.processed_data_file = processed_data_file
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = Sequential()
        self.X_test_vect = X_test_vect
        self.y_test = y_test

    def __preprocess_for_lstm(self, df):        
        def padding_sequences(encoded_reviews, sequence_length):
            features = np.zeros((len(encoded_reviews), sequence_length), dtype=int)
            for i, review in enumerate(encoded_reviews):
                if len(review) != 0:
                    features[i, -len(review):] = np.array(review)[:sequence_length]
            return features

        texts = df['cleaned_review']
        # Create the tokenizer object and fit it to the texts
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)

        # Convert texts to sequences of integers
        sequences = tokenizer.texts_to_sequences(texts)

        # Get the vocabulary size for future use
        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

        # Pad sequences to the same length
        max_length = max(len(seq) for seq in sequences)  # Use max length if not provided
        padded_sequences = padding_sequences(sequences, max_length)

        return padded_sequences, vocab_size

    def train(self):
        df = Preprocessor.preprocess(self.raw_data, self.processed_data_file)
        y = df['sentiment'].map({'positive': 1, 'negative': 0})  # This maps text labels to binary
        
        X_train, X_test, y_train, self.y_test = train_test_split(df, y, test_size=0.2, random_state=42)
        
        X_train_vect, vocab_size_train = self.__preprocess_for_lstm(X_train)
        self.X_test_vect, _ = self.__preprocess_for_lstm(X_test)
        
        self.model.add(Embedding(input_dim=vocab_size_train, output_dim=self.embedding_dim))
        self.model.add(LSTM(units=self.lstm_units, return_sequences=False))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(units=self.dense_units, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification
        
        adam = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        
        print(self.model.summary())

        checkpoint = ModelCheckpoint(
            'lstm_nn_testing.keras',
            monitor='accuracy',
            save_best_only=True,
            verbose=1
        )

        print("Beginning Training")
        self.model.fit(X_train_vect, y_train, self.batch_size, epochs=self.epochs, callbacks=[checkpoint])

    def test(self):
        y_pred = self.model.predict(self.X_test_vect, batch_size=self.batch_size)
        
        # Convert probabilities to binary labels
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(self.y_test, y_pred_binary)
        print('Accuracy: {:.2f}%'.format(accuracy * 100))
