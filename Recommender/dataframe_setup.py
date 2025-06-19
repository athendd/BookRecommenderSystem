import pandas as pd
import numpy as np

"""
Obtains all three datasets (books, ratings, and users), turns
each of them into dataframes, and then removes all NA values 
from the dataframes along with other data preprocessing
Returns: books, ratings, and users dataframes
"""
def get_dataframes(sample_size = 0):
    df_books = pd.read_csv('bookdataset/Books.csv')

    #Drop unnecessary columns from books dataframe
    df_books = df_books.drop(columns = ['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

    df_users = pd.read_csv('bookdataset/Users.csv')

    df_ratings = pd.read_csv('bookdataset/Ratings.csv')

    #Remove all rows with NA values from users dataframe
    df_users = df_users.dropna()

    #Remove all rows with NA values from books dataframe
    df_books = df_books.dropna()
    
    #Remove all rows with NA values from ratings dataframe
    df_ratings = df_ratings.dropna()
    
    #Remove all duplicate rows in each dataframe
    df_users = df_users.drop_duplicates(keep = False)
    df_books = df_books.drop_duplicates(keep = False)
    df_ratings = df_ratings.drop_duplicates(keep = False)
    
    #Remove all duplicate ISBN from books dataframe
    df_books = check_for_column_duplicates(df_books, 'ISBN')
    
    #Remove all duplicate User IDs from the users dataframe
    df_users = check_for_column_duplicates(df_users, 'User-ID')
    
    #Create a Book-ID column for each ISBN in both books and ratings dataframes
    df_books['Book-ID'] = pd.factorize(df_books['ISBN'])[0] + 1 
    df_ratings['Book-ID'] = pd.factorize(df_ratings['ISBN'])[0] + 1
    
    #Update User-ID column in ratings dataframe to start at 0
    # df_ratings['New-User-ID'] = pd.factorize(df_ratings['User-ID'])[0] + 1 
    
    if sample_size > 0:
        df_ratings = resample_dataframe(sample_size, df_ratings)
        df_ratings = df_ratings.reset_index()
        
        
    return df_books, df_ratings, df_users

"""
Removes all duplicate values from a given column in a dataframe
"""
def check_for_column_duplicates(df, col):
    df = df.drop_duplicates(subset = [col], keep = False)
    
"""
Resamples ratings dataframe to a given sample size that makes it easier to compute
"""
def resample_dataframe(sample_size, df_ratings):
    #Sampling books and users IDs for faster computation
    sample_user_ids = np.random.choice(df_ratings['User-ID'].unique(), size=sample_size, replace=False)
    sample_book_ids = np.random.choice(df_ratings['Book-ID'].unique(), size=sample_size, replace=False)

    #Filter ratings to only include the sampled users and books
    df_ratings_sampled = df_ratings[
        df_ratings['User-ID'].isin(sample_user_ids) &
        df_ratings['Book-ID'].isin(sample_book_ids)
    ]
    
    return df_ratings_sampled
    