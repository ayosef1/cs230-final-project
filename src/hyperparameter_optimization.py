import optuna
from lstm_nn import LSTMClassifier
import pandas as pd

def optimize_hyperparameters(raw_data, processed_data_file, n_trials=20):
    # Objective function for Optuna
    def objective(trial):
        # Sample hyperparameters
        embedding_dim = trial.suggest_int("embedding_dim", 50, 150, step=25)
        lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.1)
        dense_units = trial.suggest_int("dense_units", 32, 128, step=32)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-2)

        classifier = LSTMClassifier(
            raw_data=raw_data,
            processed_data_file=processed_data_file,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        _, history = classifier.train()

        val_accuracy = history.history["val_accuracy"][-1]

        return val_accuracy
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    study.trials_dataframe().to_csv("optuna_trials_metadata.csv", index=False)
    best_trial = study.best_trial
    print("Best Trial:")
    print(f"  Accuracy: {best_trial.value:.4f}")
    print(f"  Params: {best_trial.params}")

def main():
    raw_data = 'data/imdb_dataset.csv'
    processed_data_file = 'data/imdb_dataset.pkl'

    optimize_hyperparameters(raw_data, processed_data_file, 1)

if __name__ == '__main__':
    main()