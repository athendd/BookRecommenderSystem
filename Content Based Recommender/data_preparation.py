import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

"""
535 different genres
Create a new column that is a combination of budget and revenue
"""

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
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    
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

    #Clean text for overview
    df['overview'] = df['overview'].apply(clean_text)
    
    #Adding weights to recommendations
    #df['combined_text'] = df['overview'] * 2 + ' ' + df['genres'] + ' ' + df['keywords']

    #Create a column made up of keywords, genre, and overview
    df['combined_text'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords'] + ' ' + df['director']

    return df