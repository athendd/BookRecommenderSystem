from dataframe_setup import get_dataframes
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import normalize

"""
Collaborative Filtering recommends books based on
the user's preference 
"""
def recommend_books(user_id, data_matrix, k = 10):
    data_matrix = data_matrix.tocsr()

    df_user = df_ratings[df_ratings['User-ID'] == user_id]
            
    #Finds the given user's highest rated book
    book_id = df_user[df_user['Book-Rating'] == max(df_user['Book-Rating'])]['Book-ID'].iloc[0]
    
    book_titles = dict(zip(df_books['Book-ID'], df_books['Book-Title']))
    
    similar_books = find_similar_books_using_knn(book_id, data_matrix, k) 
    
    for similar_book in similar_books:
        if similar_book in book_titles:
            print(book_titles[similar_book])
            
def find_similiar_books_using_svd(book_id, data_matrix, k):
    svd = TruncatedSVD(n_components = 50, random_state = 42)
    latent_matrix = svd.fit_transform(data_matrix)
    
    #Normalize the matrix
    latent_matrix = normalize(latent_matrix)
    
    book_vector = latent_matrix[book_id - 1]
    
    sims = latent_matrix.dot(book_vector)
    top_k_books = np.argsort(-sims)[1:k+1]
    
    return top_k_books
    
def find_similar_books_using_knn(book_id, data_matrix, k, metric = 'cosine'):
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

def train_and_test(df_ratings):
    test_data = []
    train_data = []
    for user_id in df_ratings['User-ID'].unique():
        user_ratings = df_ratings[df_ratings['User-ID'] == user_id]
        if len(user_ratings) > 1:
            test_rating = user_ratings.sample(1)
            train_rating = user_ratings.drop(test_rating.index)
            test_data.append(test_rating)
            train_data.append(train_rating)
            
    df_train = pd.concat(train_data)
    df_test = pd.concat(test_data)
    
    precision = precision_at_k(df_train, df_test, 10)
    recall = recall_at_k(df_train, df_test, 10)
    avg_precision = average_precision_at_k(df_train, df_test, 10)
    
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Average Precision: {avg_precision}')

def precision_at_k(recommended, actual, k):
    return len(set(recommended[:k]) & set(actual)) / k

def recall_at_k(recommended, actual, k):
    return len(set(recommended[:k]) & set(actual)) / len(actual)

def average_precision_at_k(recommended, actual, k):
    score = 0.0
    hits = 0
    for i in range(k):
        if recommended[i] in actual:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)    

if __name__ == 'main':

    df_books, df_ratings, df_users = get_dataframes(sample_size = 10000)

    #Get the number of ratings, books, and users
    n_ratings = len(df_ratings)
    n_books = df_ratings['Book-ID'].nunique()
    n_users = df_ratings['User-ID'].nunique()

    #Create mappings from User-ID and Book-ID to matrix indices
    user_to_index = {user_id: index for index, user_id in enumerate(df_ratings['User-ID'].unique())}
    book_to_index = {book_id: index for index, book_id in enumerate(df_ratings['Book-ID'].unique())}

    data_matrix = lil_matrix((n_users, n_books))

    for _, row in df_ratings.iterrows():
        user_idx = user_to_index[row['User-ID']]
        book_idx = book_to_index[row['Book-ID']]
        data_matrix[user_idx, book_idx] = row['Book-Rating']

    user_id = df_ratings.iloc[0]['User-ID']

    user_ids = df_ratings['User-ID'].tolist()
    
    if user_id in user_ids:
        recommend_books(user_id, data_matrix)


