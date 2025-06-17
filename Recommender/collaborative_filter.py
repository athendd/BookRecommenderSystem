from dataframe_setup import get_dataframes
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

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

df_books, df_ratings_sampled, df_users = get_dataframes(sample_size = 5000)

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

"""
Each row is a user and each column is a book and the value is the predicted 
rating that user would give that book (opposite for book prediciton)
"""
print(user_prediction)
print(book_prediction)