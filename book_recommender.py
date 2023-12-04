import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import kaggle
from pydantic import BaseModel
from typing import List, Optional

import warnings

from book_identifier import get_chatgpt_recommendations, identify_book_from_prompt
warnings.filterwarnings('ignore')

# Paths to the datasets
MERGED_DATASER_PATH     = 'datasets/merged.csv'
BOOKS_DATASET_PATH      = 'datasets/Books.csv'
RATINGS_DATASET_PATH    = 'datasets/Ratings.csv'
USERS_DATASET_PATH      = 'datasets/Users.csv'

class Recommendation(BaseModel):
    """
    A model for a book recommendation.

    Attributes:
        title (str): The title of the book.
        author (str): The author of the book.
        summary (str): A summary of the book.
        rating (float, optional): The rating of the book. Defaults to None.
    """
    title: str
    author: str
    summary: str
    rating: Optional[float] = None


class RecommendationRequest(BaseModel):
    """
    A model for a book recommendation request.

    Attributes:
        prompt (str): The prompt for the recommendation.
        number_of_recommendations (int, optional): The number of recommendations to return. Defaults to 10.
    """
    prompt: str
    number_of_recommendations: int = 10


class RecommendationResponse(BaseModel):
    """
    A model for a book recommendation response.

    Attributes:
        title (str): The title of the identified book.
        author (str, optional): The author of the identified book. Defaults to None.
        summary (str, optional): A summary of the identified book. Defaults to None.
        title_alternative (str, optional): An alternative title of the identified book. Defaults to None.
        rating (float, optional): The rating of the identified book. Defaults to None.
        chatGPT_rec (bool, optional): Whether the recommendation is from ChatGPT. Defaults to False.
        recommendations (List[Recommendation]): The list of book recommendations.
    """
    title: str
    author: Optional[str] = None
    summary: Optional[str] = None
    title_alternative: Optional[str] = None
    rating: Optional[float] = None
    chatGPT_rec: bool = False
    recommendations: List[Recommendation]


def refresh_book_recommendation_datasets():
    """
    Refresh the book recommendation datasets.

    This function downloads the latest versions of the book recommendation datasets from Kaggle.

    Returns:
        bool: True if the datasets were saved successfully, False otherwise.
    """
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('arashnic/book-recommendation-dataset', path='datasets', unzip=True)
    
    return save_merged_dataset()

def save_merged_dataset():
    """
    Save the merged dataset.

    This function should merge the downloaded datasets and save the result.

    Returns:
        bool: True if the merged dataset was saved successfully, False otherwise.
    """
    try:
        ratings = pd.read_csv(RATINGS_DATASET_PATH, encoding='cp1251', sep=',')
        ratings = ratings[ratings['Book-Rating']!=0]
    except Exception as e:
        print('ERROR: ', e)
        return False

    try:
        books = pd.read_csv(BOOKS_DATASET_PATH, encoding='cp1251', sep=',', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print('ERROR: ', e)
        return False
    
    try:
        dataset = pd.merge(ratings, books, on=['ISBN'])
        dataset_lowercase = dataset.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

        dataset_lowercase['Book-Title'] = dataset_lowercase['Book-Title'].str.replace('[^\w\s]','', regex=True)

        dataset_lowercase.to_csv(MERGED_DATASER_PATH, index=False)
    except Exception as e:
        print('ERROR: ', e)
        return False
    
    return True

def get_merged_dataset():
    """
    Get the merged dataset.

    This function reads the merged dataset from the CSV file and returns it as a pandas DataFrame.

    Returns:
        DataFrame: The merged dataset.
    """
    return pd.read_csv(MERGED_DATASER_PATH, low_memory=False)

def get_books_dataset():
    """
    Get the books dataset.

    This function reads the books dataset from the CSV file, converts all string columns to lowercase, removes special characters from the 'Book-Title' column, and returns it as a pandas DataFrame.

    Returns:
        DataFrame: The books dataset.
    """
    dataset = pd.read_csv(BOOKS_DATASET_PATH, encoding='cp1251', sep=',', on_bad_lines='skip', low_memory=False).apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    dataset['Book-Title'] = dataset['Book-Title'].str.replace('[^\w\s]','', regex=True)
    return dataset

def get_ratings_dataset():
    """
    Get the ratings dataset.

    This function reads the ratings dataset from the CSV file, converts all string columns to lowercase, removes special characters from the 'Book-Title' column, and returns it as a pandas DataFrame.

    Returns:
        DataFrame: The ratings dataset.
    """
    dataset = pd.read_csv(RATINGS_DATASET_PATH, encoding='cp1251', sep=',').apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    dataset['Book-Title'] = dataset['Book-Title'].str.replace('[^\w\s]','', regex=True)
    return dataset

def get_users_dataset():
    """
    Get the users dataset.

    This function reads the users dataset from the CSV file and returns it as a pandas DataFrame.

    Returns:
        DataFrame: The users dataset.
    """
    return pd.read_csv(USERS_DATASET_PATH, encoding='cp1251', sep=',')

def get_book_by_id(book_id):
    """
    Get a book by its ID.

    This function retrieves a book from the merged dataset using its ID.

    Args:
        book_id (str): The ID of the book.

    Returns:
        Series: The book data.
    """
    dataset = get_merged_dataset()
    return dataset[dataset['ISBN'] == str(book_id)].iloc[0]

def get_book_by_title(book_title):
    """
    Get a book by its title.

    This function retrieves a book from the merged dataset using its title.

    Args:
        book_title (str): The title of the book.

    Returns:
        Series: The book data.
    """
    dataset = get_merged_dataset()
    return dataset[dataset['Book-Title'].str.contains(book_title)].iloc[0]

def get_book_by_author(author):
    """
    Get a book by its author.

    This function retrieves a book from the merged dataset using its author.

    Args:
        author (str): The author of the book.

    Returns:index
        Series: The book data.
    """
    dataset = get_merged_dataset()
    return dataset[dataset['Book-Author'].str.contains(author)].iloc[0]
    
def similar(a, b):
    """
    Calculate the similarity between two strings.

    Args:
        a (str): The first string.
        b (str): The second string.

    Returns:
        float: The similarity ratio between the two strings.
    """
    return SequenceMatcher(None, a, b).ratio()

def get_book_title_variants(original_book_title, original_book_author=None):
    """
    Get variants of a book title.

    Args:
        original_book_title (str): The original book title.
        original_book_author (str, optional): The original book author. Defaults to None.

    Returns:
        list: A list of book title variants.
    """
    author_title_match_threshold = 0.6
    title_match_threshold = 0.5

    books_dataset = get_books_dataset()

    original_book_title = original_book_title.lower()
    original_book_author = original_book_author.lower() if original_book_author else None

    if not books_dataset.empty:

        if original_book_author:
            book_author = original_book_author
        else:
            # trying to guess the author from the title
            books_dataset = books_dataset[books_dataset['Book-Author'].notna()]

            potencial_authors = books_dataset[books_dataset['Book-Title'].notna() & (books_dataset['Book-Title'].str.contains(original_book_title))]['Book-Author']

            if potencial_authors.empty:

                potencial_authors = books_dataset[books_dataset['Book-Title'].notna() & (books_dataset['Book-Title'].apply(lambda title: similar(original_book_title, title) > author_title_match_threshold))]['Book-Author']

                if potencial_authors.empty:

                    return []

            book_author = pd.Series(potencial_authors).mode()[0]
            print('Book author should be', book_author)

        books_dataset = books_dataset[books_dataset['Book-Author'].notna() & (books_dataset['Book-Author'].str.contains(book_author, case=False))]

        title_variants_threshold = books_dataset[books_dataset['Book-Title'].notna() & (books_dataset['Book-Title'].apply(lambda title: similar(original_book_title, title) > title_match_threshold))]['Book-Title'].unique().tolist()

        title_variants_contains = books_dataset[books_dataset['Book-Title'].notna() & (books_dataset['Book-Title'].str.contains(original_book_title, case=False))]['Book-Title'].unique().tolist()

        return list(set(title_variants_threshold + title_variants_contains))
    
    return []

def get_book_title_variants_by_author(original_book_author):
    """
    Get variants of a book title by author. !!!TBD!!!

    Args:
        original_book_author (str): The original book author.

    Returns:
        list: A list of book title variants.
    """
    author_match_threshold = 0.6

    books_dataset = get_books_dataset()

    original_book_author = original_book_author.lower()

    if not books_dataset.empty:

        return books_dataset[books_dataset['Book-Author'].notna() & (books_dataset['Book-Author'].str.contains(original_book_author, case=False))]['Book-Title'].unique().tolist()
    
    return []

def get_book_recommendations_by_title_from_dataset(book_title_variants, number_of_recommendations=10, top=True, number_of_ratings=8):
    """
    Get book recommendations by title from dataset.

    Args:
        book_title_variants (list): A list of book title variants.
        number_of_recommendations (int, optional): The number of recommendations to return. Defaults to 10.
        top (bool, optional): Whether to return the top recommendations. Defaults to True.
        number_of_ratings (int, optional): The minimum number of ratings a book must have. Defaults to 8.

    Returns:
        DataFrame: A DataFrame of book recommendations.
    """
    dataset = get_merged_dataset()

    # empty lists
    book_titles = []
    book_authors = []
    correlations = []
    avgrating = []
        
    for title_variant in book_title_variants:
    
        readers = np.unique(dataset['User-ID'][dataset['Book-Title']==title_variant].tolist())
        
        if readers.size == 0:
            print('No readers for book: ', title_variant)
            continue

        # final dataset
        books_of_readers = dataset[(dataset['User-ID'].isin(readers))]

        # Number of ratings per other books in dataset
        number_of_rating_per_book = books_of_readers.groupby(['Book-Title']).agg('count').reset_index()

        # select only books which have actually higher number of ratings than threshold
        books_to_compare = number_of_rating_per_book['Book-Title'][number_of_rating_per_book['User-ID'] >= number_of_ratings]
        books_to_compare = books_to_compare.tolist()

        ratings_data_raw = books_of_readers[['User-ID', 'Book-Rating', 'Book-Title', 'Book-Author']][books_of_readers['Book-Title'].isin(books_to_compare)]    

        # group by User and Book and compute mean
        ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()

        # reset index to see User-ID in every row
        ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()

        dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

        dataset_of_other_books = dataset_for_corr.copy(deep=False)

        try:
            dataset_of_other_books.drop([title_variant], axis=1, inplace=True)
        except Exception as e:
            print('WARNING: ', e)
            continue

        print('Computing correlations for book: ', title_variant)
        # corr computation
        for other_book_title in list(dataset_of_other_books.columns.values):
            
            book_titles.append(other_book_title)
            
            book_authors.append(books_of_readers[books_of_readers['Book-Title'] == other_book_title]['Book-Author'].values[0])

            correlations.append(dataset_for_corr[title_variant].corr(dataset_of_other_books[other_book_title]))
            
            tab = ratings_data_raw[ratings_data_raw['Book-Title']==other_book_title].groupby(ratings_data_raw['Book-Title'])['Book-Rating'].mean()
            avgrating.append(tab.values.min())

    result_list = []
    worst_list = []
    
    # final dataframe of all correlation of each book   
    corr_book = pd.DataFrame(list(zip(book_titles, book_authors, correlations, avgrating)), columns=['book', 'author', 'corr','avg_rating'])
    corr_book.head()

    if top:
        # top 10 books with highest corr
        result_list.append(corr_book.sort_values('corr', ascending = False).head(number_of_recommendations))
        return result_list[0]
    else:
        # worst 10 books
        worst_list.append(corr_book.sort_values('corr', ascending = False).tail(number_of_recommendations))
        return worst_list[0]

def get_book_recommendations_by_title(original_book_title, original_book_author, book_title_variants, number_of_recommendations=10, top=True, number_of_ratings=8):
    """
    Get book recommendations by title.

    Args:
        original_book_title (str): The original book title.
        original_book_author (str): The original book author.
        book_title_variants (list): A list of book title variants.
        number_of_recommendations (int, optional): The number of recommendations to return. Defaults to 10.
        top (bool, optional): Whether to return the top recommendations. Defaults to True.
        number_of_ratings (int, optional): The minimum number of ratings a book must have. Defaults to 8.

    Returns:
        tuple: A tuple containing a boolean indicating whether the recommendation is from ChatGPT and a list of book recommendations.
    """
    recommendations = get_book_recommendations_by_title_from_dataset(book_title_variants, number_of_recommendations, top, number_of_ratings)

    results = []

    if recommendations['book'].size:

        for index, recommendation in recommendations.iterrows():
            prompt = f"{recommendation['book']} by {recommendation['author']}"

            result = identify_book_from_prompt(prompt)
            result['rating'] = recommendation['avg_rating']

            results.append(result)

        chatGPT_rec = False

    else:
        chatGPT_rec = True
        results = get_chatgpt_recommendations(original_book_title, original_book_author, 4)
        
    return chatGPT_rec, results
