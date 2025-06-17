from dataframe_setup import get_dataframes
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from sklearn.preprocessing import LabelEncoder

def predict(ratings, similarity, type):
    if type == 'user':
        mean_user_rating = ratings.mean(axis = 1)
        
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis = 1)]).T
    elif type == 'book':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis = 1)])
    
    else:
        pred = None
        
    return pred

df_books, df_ratings, df_users = get_dataframes()

#Sampling books and users IDs to 5000 for faster computation
sample_user_ids = np.random.choice(df_ratings['New-User-ID'].unique(), size=5000, replace=False)
sample_book_ids = np.random.choice(df_ratings['Book-ID'].unique(), size=5000, replace=False)

#Filter ratings to only include the sampled users and books
df_ratings_sampled = df_ratings[
    df_ratings['New-User-ID'].isin(sample_user_ids) &
    df_ratings['Book-ID'].isin(sample_book_ids)
]

#Encode user and book IDs to avoid sparse matrix explosion
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()

df_ratings_sampled['user_idx'] = user_encoder.fit_transform(df_ratings_sampled['New-User-ID'])
df_ratings_sampled['book_idx'] = book_encoder.fit_transform(df_ratings_sampled['Book-ID'])

n_users = df_ratings_sampled['user_idx'].nunique()
n_books = df_ratings_sampled['book_idx'].nunique()

#User-Book matrix that calculates the similarity between books and users
data_matrix = np.zeros((n_users, n_books))

for row in df_ratings_sampled.itertuples():
    data_matrix[row.user_idx, row.book_idx] = row[3] 
        
#Turns book-book and user-user similarity into arrays
user_similarity = pairwise_distances(data_matrix, metric = 'cosine')
book_similarity = pairwise_distances(data_matrix.T, metric = 'cosine')

user_prediction = predict(data_matrix, user_similarity, type = 'user')
book_prediction = predict(data_matrix, book_similarity, type = 'book')

print(user_prediction)
print(book_prediction)