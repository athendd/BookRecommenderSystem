from data_preparation import clean_dataset
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

model = SentenceTransformer('all-MiniLM-L6-v2')  

def weighted_embedding(row):
    embeddings = []
    weights = []
    
    features = {
        "overview": row['overview'],
        "genres": row['genres'],
        "keywords": row['keywords'],
        "director": row['director'],
        "tagline": row['tagline'],
        "year": row['year'],
        "language": row['original_language']
    }
    
    feature_weights = {
        "overview": 2.0,
        "genres": 1.75,
        "keywords": 1.2,
        "director": 1.5,
        "tagline": 1.0,
        "year": 0.75,
        "language": 0.5
    }

    for feature, text in features.items():
        if pd.notnull(text):
            emb = model.encode(text, normalize_embeddings=True)
            embeddings.append(emb * feature_weights[feature])
            weights.append(feature_weights[feature])

    combined = np.sum(embeddings, axis=0) / np.sum(weights)
    
    return combined
    
"""
Recommends top N movies based on cosine similarity.

movie_title: The title of the movie to get recommendations for.
df: Your DataFrame containing movie information,
                    including 'title'.
similarity_matrix: The cosine similarity matrix.
num_recommendations: The number of top recommendations to return.

Returns a series of recommended movie titles
"""
def get_movie_recommendations(movie, df, similarity_matrix, num_recommendations=10):

    #Check if the movie exists in the DataFrame
    if movie not in df['original_title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return pd.Series([])

    #Get the index of the movie
    movie_index = df[df['original_title'] == movie].index[0]

    #Get the similarity scores for the movie
    similar_movies_scores = list(enumerate(similarity_matrix[movie_index]))

    #Sort the movies based on the similarity scores in descending order
    sorted_similar_movies = sorted(similar_movies_scores, key=lambda x: x[1], reverse=True)
    
    #Remove the chosen movie
    sorted_similar_movies = [movie for movie in sorted_similar_movies if movie[0] != movie_index] 

    #Get the indices of the top N recommended movies
    top_n_movie_indices = [movie[0] for movie in sorted_similar_movies[0:num_recommendations]]

    #Return the titles of the recommended movies
    return df['title'].iloc[top_n_movie_indices]

df = clean_dataset()

embeddings = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    embeddings.append(weighted_embedding(row))

df['combined_text'] = embeddings

documents = df['combined_text'].tolist()

print('yes')

similarity_matrix = cosine_similarity(documents)

#Title of chosen movie
movie_to_recommend_for = "Avatar" 

top_10_recommendations = get_movie_recommendations(movie_to_recommend_for, df, similarity_matrix, num_recommendations=10)

if not top_10_recommendations.empty:
    print(f"\nTop 10 recommended movies for '{movie_to_recommend_for}':")
    for i, movie in enumerate(top_10_recommendations):
        print(f"{i+1}. {movie}")
