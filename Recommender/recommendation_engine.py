#Try using a system with both cotent and collaborative filtering
#Hybrid model could averag eout both models' ratings or combine the scores together
#Edit this to handle test data
from matrix_factorization_recommendation import Matrix_Factorization
import numpy as np
from dataframe_setup import get_dataframes
from sklearn.model_selection import train_test_split

df_books, df_ratings, df_users = get_dataframes(sample_size = 10000)

original_matrix = np.array(df_ratings.pivot(index = 'User-ID', columns ='Book-ID', values = 'Book-Rating').fillna(0))

num_users, num_items = original_matrix.shape
ratings = [(i, j, original_matrix[i, j]) for i in range(num_users) for j in range(num_items) if original_matrix[i, j] > 0]

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

train_matrix = np.zeros_like(original_matrix)
for i, j, r in train_data:
    train_matrix[i, j] = r
    
matrix_factorization = Matrix_Factorization(train_matrix, 40, 0.001, 0.01, iterations = 100, train_data = train_data, test_data = test_data)
matrix_factorization.train()

recall, precision = matrix_factorization.compute_recall_precision_at_k(5, 6)
print(recall)
print(precision)
