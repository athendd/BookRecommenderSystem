from dataframe_setup import get_dataframes
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix

df_books, df_ratings, df_users = get_dataframes()

atings_train, ratings_test = train_test_split(df_ratings, test_size = 0.2, random_state = 42)

#Get number of users 
n_users = df_ratings['New-User-ID'].unique().shape[0]

#Get number of movies
n_movies = df_ratings['Book-ID'].unique().shape[0]

#User-Movie matrix that calculates the similarity between movies and users
data_matrix = lil_matrix((n_users, n_movies))

for line in df_ratings.itertuples():
    data_matrix[line[5]-1, line[4]-1] = line[3]  
    

