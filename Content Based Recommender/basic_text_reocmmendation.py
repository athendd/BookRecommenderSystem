from data_preparation import clean_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

#Use BERT instead of TfidfVectorizer

"""
Recommends top N movies based on cosine similarity.

Args:
    movie_title (str): The title of the movie to get recommendations for.
    df (pd.DataFrame): Your DataFrame containing movie information,
                        including 'title'.
    similarity_matrix (np.array): The cosine similarity matrix.
    num_recommendations (int): The number of top recommendations to return.

Returns: A Series of recommended movie titles
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

documents = df['combined_text'].tolist()

#vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 5000)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

document_embeddings = []

for document in tqdm(documents):
    inputs = tokenizer(document, return_tensors = 'pt', padding = True, truncation = True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    #cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    cls_embedding = (sum_embeddings / sum_mask).squeeze().numpy()
    document_embeddings.append(cls_embedding)

#tfidf_matrix = vectorizer.fit_transform(documents)

#similarity_matrix = cosine_similarity(tfidf_matrix)

"""
Way to incorporate numerical data

scaler = MinMaxScaler()
numeric_features = scaler.fit_transform(df[['total_profit', 'popularity']])
final_embeddings = [np.concatenate((bert_emb, numeric_features[i])) for i, bert_emb in enumerate(document_embeddings)]
"""

similarity_matrix = cosine_similarity(document_embeddings)


#Title of chosen movie
movie_to_recommend_for = "Avatar" 

top_10_recommendations = get_movie_recommendations(movie_to_recommend_for, df, similarity_matrix, num_recommendations=10)

if not top_10_recommendations.empty:
    print(f"\nTop 10 recommended movies for '{movie_to_recommend_for}':")
    for i, movie in enumerate(top_10_recommendations):
        print(f"{i+1}. {movie}")
