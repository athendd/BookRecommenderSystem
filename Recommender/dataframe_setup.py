import pandas as pd

"""
Obtains all three datasets (books, ratings, and users), turns
each of them into dataframes, and then removes all NA values 
from the dataframes along with other data preprocessing
Returns: books, ratings, and users dataframes
"""
def get_dataframes():
    df_books = pd.read_csv('bookdataset/Books.csv')

    #Drop unnecessary columns from books dataframe
    df_books = df_books.drop(columns = ['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

    df_users = pd.read_csv('bookdataset/Users.csv')

    df_ratings = pd.read_csv('bookdataset/Ratings.csv')

    #Remove all rows without an age value from users dataframe
    df_users = df_users.dropna()

    #Remove all rows with NA values from books dataframe
    df_books = df_books.dropna()
    
    #Create a Book-ID column for each ISBN in both books and ratings dataframes
    df_books['Book-ID'] = pd.factorize(df_books['ISBN'])[0] + 1 
    df_ratings['Book-ID'] = pd.factorize(df_ratings['ISBN'])[0] + 1
    
    #Update User-ID column in ratings dataframe to start at 0
    df_ratings['New-User-ID'] = pd.factorize(df_ratings['User-ID'])[0] + 1 
        
    return df_books, df_ratings, df_users
    
    