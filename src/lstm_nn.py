import numpy as np
import matplotlib.pyplot as plt
from preprocessor import Preprocessor
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class LSTMClassifier:
    def __init__(self,
                 raw_data, 
                 processed_data_file,
                 checkpoint=None,
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
        self.checkpoint = checkpoint
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

        # Map the label column to binary labels
        y = df['sentiment'].map({'positive': 1, 'negative': 0})

        # Drop the label column from the features
        X = df.drop(columns=['sentiment'])

        # Split into training and temporary sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

        # Split the temporary set into dev and test sets
        X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Tokenize and process the training set
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train['cleaned_review'])  # Fit tokenizer on training data only

        # Convert texts to sequences
        X_train_sequences = tokenizer.texts_to_sequences(X_train['cleaned_review'])
        X_dev_sequences = tokenizer.texts_to_sequences(X_dev['cleaned_review'])
        X_test_sequences = tokenizer.texts_to_sequences(X_test['cleaned_review'])

        # Determine max length from training data
        max_length = max(len(seq) for seq in X_train_sequences)

        # Pad all sequences to the same length
        X_train_padded = padding_sequences(X_train_sequences, max_length)
        X_dev_padded = padding_sequences(X_dev_sequences, max_length)
        X_test_padded = padding_sequences(X_test_sequences, max_length)

        # Get vocabulary size
        vocab_size = len(tokenizer.word_index) + 1  # Add 1 for reserved 0 index

        # Return all processed data and labels
        return (X_train_padded, y_train, 
                X_dev_padded, y_dev, 
                X_test_padded, y_test, 
                vocab_size)

    def train(self):
        df = Preprocessor.preprocess(self.raw_data, self.processed_data_file)

        # Preprocess data and split into training, dev, and test sets
        X_train_vect, y_train, X_dev_vect, y_dev, self.X_test_vect, self.y_test, vocab_size_train = self.__preprocess_for_lstm(df)
        
        self.model.add(Embedding(input_dim=vocab_size_train, output_dim=self.embedding_dim))
        self.model.add(LSTM(units=self.lstm_units, return_sequences=False))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(units=self.dense_units, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification
        
        adam = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        
        print(self.model.summary())

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        if self.checkpoint:
            checkpoint = ModelCheckpoint(
                self.checkpoint,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            callbacks = [checkpoint, early_stopping]
        else:
            callbacks = [early_stopping]


        print("Beginning Training")
        history = self.model.fit(
            X_train_vect, 
            y_train, 
            self.batch_size, 
            epochs=self.epochs,
            validation_data=(X_dev_vect, y_dev),
            callbacks=callbacks
        )
        
        if self.checkpoint:
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.legend()
            plt.show()

        return self.model, history

    def test(self):
        y_pred = self.model.predict(self.X_test_vect, batch_size=self.batch_size)
        
        # Convert probabilities to binary labels
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(self.y_test, y_pred_binary)
        print('Testing Accuracy: {:.2f}%'.format(accuracy * 100))
