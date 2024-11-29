from src.preprocess import preprocess


def main():
    
    filename = 'data/imdb_dataset.csv'
    processed_df = preprocess(filename)
    
    print(processed_df.head())
    
    
if __name__ == "__main__":
    main()


