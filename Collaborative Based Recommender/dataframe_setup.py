import pandas as pd
import numpy as np

"""
Obtains all three datasets (books, ratings, and users), turns
each of them into dataframes, and then removes all NA values 
from the dataframes along with other data preprocessing
Returns: books, ratings, and users dataframes
"""
def get_dataframes(sample_size = 0):
    df_books = pd.read_csv('bookdataset/Books.csv', low_memory = False)
    
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
    df_users = df_users.drop_duplicates(keep = 'first')
    df_books = df_books.drop_duplicates(keep = 'first')
    df_ratings = df_ratings.drop_duplicates(keep = 'first')
    
    #Remove all duplicate ISBN from books dataframe
    df_books = check_for_column_duplicates(df_books, 'ISBN')
        
    #Remove all duplicate User IDs from the users dataframe
    df_users = check_for_column_duplicates(df_users, 'User-ID')
    
    #Create a Book-ID column for each ISBN in both books and ratings dataframes
    df_books['Book-ID'] = pd.factorize(df_books['ISBN'])[0] + 1 
    df_ratings['Book-ID'] = pd.factorize(df_ratings['ISBN'])[0] + 1
    
    if sample_size > 0 and sample_size < len(df_ratings):
        df_ratings = resample_dataframe(sample_size, df_ratings)
        df_ratings = df_ratings.reset_index()
        
        
    return df_books, df_ratings, df_users

"""
Removes all duplicate values from a given column in a dataframe
"""
def check_for_column_duplicates(df, col):
    df = df.drop_duplicates(subset = [col], keep = 'first')
    
    return df
    
"""
Resamples ratings dataframe to a given sample size that makes it easier to compute
"""
def resample_dataframe(sample_size, df_ratings, min_user_ratings = 20, min_book_ratings = 20):
    user_counts = df_ratings['User-ID'].value_counts()
    
    #Finds users who have rated at least the min_user_ratings books
    active_users = user_counts[user_counts >= min_user_ratings].index

    #Dataframe containing only users that have rated equal to or over the min_user_ratings
    df_active = df_ratings[df_ratings['User-ID'].isin(active_users)]

    book_counts = df_active['Book-ID'].value_counts()
    
    #Finds books that have been rated equal to or over the min_book_ratings
    popular_books = book_counts[book_counts >= min_book_ratings].index

    #Dataframe containg only books that have been rated equal to or over the min_book_ratings
    df_filtered = df_active[df_active['Book-ID'].isin(popular_books)]

    #Randomly samples users from the dataset
    sampled_users = np.random.choice(df_filtered['User-ID'].unique(), size=min(sample_size, len(df_filtered['User-ID'].unique())), replace=False)
    sampled_books = np.random.choice(df_filtered['Book-ID'].unique(), size=min(sample_size, len(df_filtered['Book-ID'].unique())), replace=False)

    df_sampled = df_filtered[
        df_filtered['User-ID'].isin(sampled_users) &
        df_filtered['Book-ID'].isin(sampled_books)
    ].copy()

    #Dictionaries that maps the original book and user ids to the ones in df_sampled
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(df_sampled['User-ID'].unique())}
    book_id_map = {old_id: new_id for new_id, old_id in enumerate(df_sampled['Book-ID'].unique())}

    df_sampled['User-ID'] = df_sampled['User-ID'].map(user_id_map)
    df_sampled['Book-ID'] = df_sampled['Book-ID'].map(book_id_map)

    return df_sampled
