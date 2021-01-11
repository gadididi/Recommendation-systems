import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(df_book_id, predicted_ratings_row, data_matrix_row, items, k=5):
    nan_indexes = data_matrix_row[data_matrix_row.isnull().any(1)]
    l = nan_indexes.index.tolist()
    predicted_ratings_unrated = predicted_ratings_row.iloc[l]
    # print(predicted_ratings_unrated)

    idx = predicted_ratings_unrated.nlargest(k, "rating")
    original_book_id = items['book_id']
    # Return top k movies
    return items['book_id'].iloc[idx.index.tolist()]


class CollaborativeFiltering:

    def __init__(self, ratings):
        self.ratings = ratings
        self.pred_table = {'cosine': None, 'euclidean': None, 'jaccard': None}
        self.number_of_users = None
        self.number_of_items = None
        self.data_matrix = None

    def build_CF_prediction_matrix(self, sim):
        if self.pred_table[sim] is not None:
            return self.pred_table[sim], self.data_matrix
        self.number_of_users = self.ratings.user_id.unique().shape[0]
        self.number_of_items = self.ratings.book_id.unique().shape[0]
        rating_copy = self.ratings.copy()
        rating_copy.book_id, uniques = pd.factorize(self.ratings.book_id)
        self.data_matrix = self.create_data_matrix(rating_copy)
        mean_user_rating = np.nanmean(self.data_matrix, axis=1).reshape(-1, 1)
        ratings_diff = (self.data_matrix - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0
        user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
        self.pred_table[sim] = mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
            [np.abs(user_similarity).sum(axis=1)]).T
        return self.pred_table[sim], self.data_matrix

    def create_data_matrix(self, rating_copy):
        data_matrix = np.empty((self.number_of_users, self.number_of_items))
        data_matrix[:] = np.nan
        for line in rating_copy.itertuples():
            user = line[1] - 1
            book = line[2] - 1
            rating = line[3]
            data_matrix[user, book] = rating
        return data_matrix

    def get_CF_recommendation(self, user_id, k):
        df_book_id = (self.ratings[['book_id']].copy())
        new_id, uni = pd.factorize(self.ratings.book_id)
        df_1 = pd.DataFrame(new_id, columns=['new_book_id'])
        df_book_id = df_book_id.join(df_1)
        pred_matrix, data_matrix = self.build_CF_prediction_matrix('cosine')
        user = user_id - 1
        predicted_ratings_row = pred_matrix[user]
        data_matrix_row = data_matrix[user]
        data_matrix_row = pd.DataFrame(data_matrix_row, columns=['rating'])
        predicted_ratings_row = pd.DataFrame(predicted_ratings_row, columns=['rating'])
        print(get_recommendations(df_book_id, predicted_ratings_row, data_matrix_row, self.ratings, k=k))
