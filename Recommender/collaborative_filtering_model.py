from dataframe_setup import get_dataframes
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

"""
Collaborative Filtering recommends books based on
the user's preference 
"""
def recommend_books(user_id, data_matrix, k = 10):
    df_user = df_ratings[df_ratings['New-User-ID'] == user_id]
            
    #Finds the given user's highest rated book
    book_id = df_user[df_user['Book-Rating'] == max(df_user['Book-Rating'])]['Book-ID'].iloc[0]
    
    book_titles = dict(zip(df_books['Book-ID'], df_books['Book-Title']))
    
    similar_books = find_similar_books(book_id, data_matrix, k) 
    
    for similar_book in similar_books:
        if similar_book in book_titles:
            print(book_titles[similar_book])
    
def find_similar_books(book_id, data_matrix, k, metric = 'cosine'):
    neighbors = []
        
    book_vector = data_matrix[book_id - 1]
    
    k += 1
    knn = NearestNeighbors(n_neighbors = k, algorithm = 'brute', metric = metric)
    knn.fit(data_matrix)
    book_vector = book_vector.reshape(1, -1)
    
    neighbor = knn.kneighbors(book_vector, return_distance = False)
    
    for i in range(0, k):
        n = neighbor.item(i)
        neighbors.append(n-1)
    neighbors.pop(0)
    
    return neighbors

df_books, df_ratings, df_users = get_dataframes(sample_size = 10000)

#Get the number of ratings, books, and users
n_ratings = len(df_ratings)
n_books = df_ratings['Book-ID'].nunique()
n_users = df_ratings['New-User-ID'].nunique()

#1149780 for ratings, 340556 for books, 105283 for users

data_matrix = np.zeros((n_users, n_books))

for row in df_ratings.itertuples():
    data_matrix[row.user_idx, row.book_idx] = row[3] 

user_id = df_ratings.iloc[0]['New-User-ID']

user_ids = df_ratings['New-User-ID'].tolist()

if user_id in user_ids:
    recommend_books(user_id, data_matrix)


