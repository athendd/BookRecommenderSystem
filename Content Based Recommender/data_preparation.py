import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
535 different genres
Create a new column that is a combination of budget and revenue
"""

nlp = spacy.load("en_core_web_sm")    
stop_words = set(stopwords.words('english'))

def clean_text(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    html_pattern = r'<.*?>'

    #Lowercase the text
    text = text.lower()
    
    #Remove unncessary spaces
    text = re.sub(" +", ' ', text)
    
    #Remove urls from the text 
    text = url_pattern.sub(r'', text)
    
    #Remove html from the text
    text = re.sub(html_pattern, '', text)
    
    #Removes all characters that are not letters, numbers, apostropheres and hyphens
    text = re.sub("[^a-zA-Z0-9'-]", " ", text)
    
    doc = nlp(text)
    filtered_and_lemmatized_words = []

    #Iterate through the tokens 
    for token in doc:
        #Check if the token is not a stop word and is not punctuation/whitespace
        if not token.is_stop and not token.is_punct and not token.is_space:
            filtered_and_lemmatized_words.append(token.lemma_)

    text = " ".join(filtered_and_lemmatized_words)
    
    return text

def clean_dataset():
    df = pd.read_csv(r'Movie Dataset\movie_dataset.csv')

    #Remove uncessary columns 
    df = df.drop(columns = ['homepage'])

    #Remove all rows with N/A values
    df = df.dropna()

    #Lower every word in genre column
    df['genres'] = df['genres'].str.lower()

    #Add a total earnings column
    df['total_profit'] = df['revenue'] - df['budget']

    #Clean text for overview
    df['overview'] = df['overview'].apply(clean_text)

    #Create a column made up of keywords, genre, and overview
    df['combined_text'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords']

    return df