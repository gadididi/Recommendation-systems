import heapq
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


# combine the users and the ratings data frames.
def combine_users_ratings():
    loc_data = pd.read_csv('users.csv', low_memory=False)
    ratings_data = pd.read_csv('ratings.csv', low_memory=False)
    rate_data = ratings_data.merge(loc_data, on="user_id", how='inner')
    return rate_data


# combine the books title and the books_df data frames according to book_id column.
def combine_books_score(books_df):
    books_data = pd.read_csv('books.csv', encoding="ISO-8859-1")
    books_names = books_data[["title", "book_id"]]
    return books_df.merge(books_names, on="book_id", how='inner')


# creates a data frame with the score of each book according to the metadata entered and
# groups the answers by the group_by entered.
def get_books_rating(metadata):
    # Change the metadata so it will include the rating's average and the number of ratings of each book.
    metadata = metadata.groupby(["book_id"]).agg(
        vote_count=pd.NamedAgg(column="rating", aggfunc='count'),
        vote_average=pd.NamedAgg(column="rating", aggfunc='mean')
    )
    # Calculate mean of vote average column
    C = metadata['vote_average'].mean()
    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata['vote_count'].quantile(0.90)
    q_books = metadata.copy().loc[metadata['vote_count'] >= m]

    # Function that computes the weighted rating of each movie
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v / (v + m) * R) + (m / (m + v) * C)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_books['score'] = q_books.apply(weighted_rating, axis=1)
    # Sort movies based on score calculated above
    q_books = q_books.sort_values('score', ascending=False)
    # print(q_books)
    return q_books.reset_index()


# returns the first k recommendation.
def get_simply_recommendation(k):
    # Load Books Metadata
    metadata = pd.read_csv('ratings.csv', low_memory=False)
    books_ratings = get_books_rating(metadata)
    books_ratings = combine_books_score(books_ratings)[["book_id", "title", "score"]]
    top_k = books_ratings.head(n=k).to_string(index=False)
    print(top_k)
    return top_k


# returns the first k recommendation according to the place entered.
def get_simply_place_recommendation(place, k):
    users_rating_combine = combine_users_ratings()
    users_rating_combine = users_rating_combine[users_rating_combine["location"] == place]
    books_ratings = get_books_rating(users_rating_combine)
    books_ratings = combine_books_score(books_ratings)[["book_id", "title", "score"]]
    top_k = books_ratings.head(n=k).to_string(index=False)
    print(top_k)
    return top_k


# changes the data frame so the ages will appear as a x0-y1 and not a number.
def update_age_df(users_rating_combine, up, down):
    users_rating_combine = users_rating_combine[users_rating_combine["age"] <= up]
    users_rating_combine = users_rating_combine[users_rating_combine["age"] >= down]
    return users_rating_combine


# returns the first k recommendation according to the age entered.
def get_simply_age_recommendation(age, k):
    num = int(age / 10)
    if age % 10 == 0:
        num -= 1
    x1 = num*10 + 1
    y0 = num*10 + 10
    users_rating_combine = combine_users_ratings()
    updated_df = update_age_df(users_rating_combine, y0, x1)
    books_ratings = get_books_rating(updated_df)
    books_ratings = combine_books_score(books_ratings)[["book_id", "title", "score"]]
    top_k = books_ratings.head(n=k).to_string(index=False)
    print(top_k)
    return top_k


# builds the similarity matrix according to the similarity index entered.
def build_CF_prediction_matrix(sim):
    ratings_pd = pd.read_csv('ratings.csv', low_memory=False)
    # calculate the number of unique users and books.
    n_users = ratings_pd.user_id.unique().shape[0]
    n_items = ratings_pd.book_id.unique().shape[0]
    # change the book_id numbers to be in a domain between the unique numbers.
    ratings_pd.book_id, uniques = pd.factorize(ratings_pd.book_id)
    # create ranking table - that table is sparse.
    data_matrix = np.empty((n_users, n_items))
    data_matrix[:] = np.nan
    for line in ratings_pd.itertuples():
        user = line[1] - 1
        book = line[2] - 1
        rating = line[3]
        data_matrix[user, book] = rating
    # calc mean.
    mean_user_rating = np.nanmean(data_matrix, axis=1).reshape(-1, 1)
    ratings_diff = (data_matrix - mean_user_rating)
    # replace nan -> 0.
    ratings_diff[np.isnan(ratings_diff)] = 0
    # calculate user x user similarity matrix.
    user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
    # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
    # Note that the user has the highest similarity to themselves.
    # user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
    # since n-k users have similarity=0, for each user only k most similar users contribute to the predicted ratings
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred, data_matrix


def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    # replace anything lower than the cut off with 0
    arr[arr < smallest] = 0
    return arr


# Function that takes in user_id as input and outputs most similar books
def convert_new_book_id_to_old(series, df_book_id):
    new_array = [None] * len(series)
    i = 0
    for x in series:
        query = "book_id==" + str(x + 1)
        df_new = df_book_id.query(query)
        new_array[i] = df_new["new_book_id"].iloc[0]
        i += 1
    return new_array


def get_recommendations(predicted_ratings_row, data_matrix_row, items, k, df_book_id):
    predicted_ratings_unrated = predicted_ratings_row[np.isnan(data_matrix_row)]
    idx = np.argsort(-predicted_ratings_unrated)
    sim_scores = idx[0:k]
    # Return top k books
    sim_scores = convert_new_book_id_to_old(sim_scores, df_book_id)
    return items['book_id'].iloc[sim_scores]


def get_top_rated(data_matrix_row, items, k, df_book_id):
    srt_idx = np.argsort(-data_matrix_row)
    # print(~np.isnan(data_matrix_row[srt_idx]))
    srt_idx_not_nan = srt_idx[~np.isnan(data_matrix_row[srt_idx])]
    srt_idx_not_nan = convert_new_book_id_to_old(srt_idx_not_nan, df_book_id)
    return items['book_id'].iloc[srt_idx_not_nan][:k]


# returns the first k recommendation according to the similarity matrix.
def get_CF_recommendation(user_id, k):
    items = pd.read_csv('ratings.csv', low_memory=False)
    df_book_id = (items[['book_id']].copy())
    new_id, uni = pd.factorize(items.book_id)
    df_1 = pd.DataFrame(new_id, columns=['new_book_id'])
    df_book_id = df_book_id.join(df_1)
    pred_matrix, data_matrix = build_CF_prediction_matrix('cosine')
    # pred_matrix, data_matrix = build_CF_prediction_matrix('euclidean')
    # pred_matrix, data_matrix = build_CF_prediction_matrix('jaccard')
    user = user_id - 1
    predicted_ratings_row = pred_matrix[user]
    data_matrix_row = data_matrix[user]

    #print("Top rated movies by test user:")
    #ans1 = get_top_rated(data_matrix_row, items, k, df_book_id)
    #print(ans1)

    print('****** test user - user_prediction ******')
    ans = combine_books_score(get_recommendations(predicted_ratings_row, data_matrix_row, items, k,
                                                  df_book_id).reset_index()[["book_id"]])[["book_id", "title"]]

    print(ans)


# """
ans1 = get_simply_recommendation(10)
print("******************************************************************************************")
ans2 = get_simply_place_recommendation("Ohio", 10)
print("******************************************************************************************")
ans3 = get_simply_age_recommendation(28, 10)
print("******************************************************************************************")
get_CF_recommendation(511, 10)
# """
