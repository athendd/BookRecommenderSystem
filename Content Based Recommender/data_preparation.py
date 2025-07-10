import re
import spacy
import pandas as pd
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  
stop_words = set(stopwords.words('english'))

def lemmatize(text):
    doc = nlp(text)
    
    return " ".join(
        [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    )

def clean_text(text):
    text = text.lower()
    #Remove urls from the text
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    
    #Remove HTML from the text
    text = re.sub(r'<.*?>', '', text)  
    
    #Removes all characters that are not letters, numbers, hyphens, or apostrophes
    text = re.sub(r"[^a-zA-Z0-9\s'-]", ' ', text)  
    text = re.sub(r"\s+", ' ', text).strip()
    
    text = lemmatize(text)
    
    return text

def clean_dataset():
    df = pd.read_csv(r'Movie Dataset\movie_dataset.csv')

    #Remove uncessary columns 
    df = df.drop(columns = ['homepage'])

    #Remove all rows with N/A values
    df = df.dropna()

    #Lower every word in genre column
    df['genres'] = df['genres'].str.lower()
    
    df['director'] = df['director'].str.lower()
    
    df['tagline'] = df['tagline'].str.lower()
    
    df['year'] = pd.to_datetime(df['release_date']).dt.year.astype(str)
            
    df['original_language'] = df['original_language'].str.lower()

    #Clean text for overview
    df['overview'] = df['overview'].apply(clean_text)

    return df
