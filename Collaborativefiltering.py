import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class CollaborativeFiltering:

    def __init__(self, ratings):
        self.ratings = ratings

    def build_CF_prediction_matrix(self, sim):
        number_of_users = self.ratings.user_id.unique().shape[0]
        number_of_items = self.ratings.book_id.unique().shape[0]
        self.ratings.book_id, uniques = pd.factorize(self.ratings.book_id)
        data_matrix = np.empty((number_of_users, number_of_items))
        data_matrix[:] = np.nan
        for line in self.ratings.itertuples():
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
        print(2)
        return pred, data_matrix
    def get_CF_recommendation(self, user_id, k):
        pass