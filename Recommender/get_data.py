import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix

df_books = pd.read_csv('bookdataset/Books.csv')

#Drop unnecessary columns from books dataframe
df_books = df_books.drop(columns = ['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

df_users = pd.read_csv('bookdataset/Users.csv')

df_ratings = pd.read_csv('bookdataset/Ratings.csv')

#Remove all rows without an age value from users dataframe
df_users = df_users.dropna()

#Remove all rows with NA values from books dataframe
df_books = df_books.dropna()

#Split the ratings dataframe into training and testing datasets
ratings_train, ratings_test = train_test_split(df_ratings, test_size = 0.2, random_state = 42)

#Get number of users 
n_users = df_ratings['User-ID'].unique().shape[0]
#Get number of movies
n_movies = df_ratings['ISBN'].unique().shape[0]

print(type())

#User-Movie matrix that calculates the similarity between movies and users
data_matrix = lil_matrix((n_users, n_movies))
for line in df_ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]  


