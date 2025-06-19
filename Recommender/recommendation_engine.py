#Look into how to improve basic collaborative_filtering_model
#Try using a system with both cotent and collaborative filtering
#Hybrid model could averag eout both models' ratings or combine the scores together
#Edit this to handl etest data

from matrix_factorization_recommendation import Matrix_Factorization
import numpy as np
from dataframe_setup import get_dataframes

df_books, df_ratings, df_users = get_dataframes(sample_size = 10000)

original_matrix = np.array(df_ratings.pivot(index = 'New-User-ID', columns ='Book-ID', values = 'Book-Rating').fillna(0))

matrix_factorization = Matrix_Factorization(original_matrix, 40, 0.001, 0.01, iterations = 100)

training_process = matrix_factorization.train()

precision,recall = matrix_factorization.compute_precision_recall_at_k(5, 8)

print(precision)
print(recall)

