import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag


#tokenizer src/set of stop words usefdl for cleaning data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(filename): #load csv into pandas df
    df = pd.read_csv(filename)
    return df

def clean_text(text): #remove html tags/special characters
    if not isinstance(text, str):
        # If text is not a string, convert it to a string
        text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize(text): #tokenize text into words
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(text):#remove stopword from text tokens
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    new_text = ' '.join(filtered_words)
    return new_text


def get_wordnet_pos(treebank_tag):
    """Converts POS tags to a format recognized by WordNetLemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_with_pos(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    tagged_words = pos_tag(words)
    lemmatized_words = []
    for token, tag in tagged_words:
        wordnet_pos = get_wordnet_pos(tag) #or wordnet.NOUN
        lemmatized_words.append(lemmatizer.lemmatize(token, pos=wordnet_pos))
    lemmatized_text =' '.join(lemmatized_words)
    return lemmatized_text

def preprocess(filename):
    """Orchestrate the loading and preprocessing of review data."""
    df = load_data(filename)
    # Apply cleaning and preprocessing steps to the 'review' column
    df['cleaned_review'] = df['review'].apply(clean_text).apply(lambda x: x.lower())  # Normalize to lowercase during cleaning
    df['cleaned_review'] = df['cleaned_review'].apply(remove_stopwords)
    df['tokens'] = df['cleaned_review'].apply(tokenize)

    
    
    df['cleaned_review'] = df['cleaned_review'].apply(lemmatize_with_pos)
    
    return df
    
    
    
if __name__ == "__main__":
    processed_data = preprocess('../data/imdb_dataset.csv')
    print(processed_data.head())
    print(processed_data['cleaned_review'].head())

