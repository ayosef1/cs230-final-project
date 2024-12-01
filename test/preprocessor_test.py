import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessor import Preprocessor

def main():
    processed_df = Preprocessor.preprocess('data/imdb_dataset.csv', 'data/imdb_dataset.pkl')
        
    print(processed_df.head())
    
    
if __name__ == "__main__":
    main()


