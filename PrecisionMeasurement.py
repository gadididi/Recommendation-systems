import math
import pandas as pd

from Collaborativefiltering import CollaborativeFiltering


class PrecisionMeasurement:
    def __init__(self, test, ratings, books, collaborative):
        self.books = books
        self.test = test
        self.ratings = ratings
        self.user_based = collaborative

    def precision_k(self, k, sim='cosine'):
        users_precision = 0
        current_test = self.pre_possessor(k)

        group_by_user = current_test.groupby('user_id').size()
        for user in group_by_user.index:
            correct = 0
            top_k = self.user_based.get_CF_recommendation(user_id=user, k=k)
            books_id_recommended = set()
            for book in top_k:
                books_id_recommended.add(book.values[0][0])
            # get all books rated by this user
            rated_by_user = self.test[self.test['user_id'] == user]
            for data in rated_by_user.values:
                if data[1] in books_id_recommended:
                    correct += 1
            users_precision += correct / k
        return users_precision / len(group_by_user.index)

    def ARHR(self, k, sim='cosine'):
        users_precision = 0
        current_test = self.pre_possessor(k)

        group_by_user = current_test.groupby('user_id').size()
        for user in group_by_user.index:
            top_k = self.user_based.get_CF_recommendation(user_id=user, k=k)
            books_id_recommended = []
            for book in top_k:
                books_id_recommended.append(book.values[0][0])
            # get all books rated by this user
            rated_by_user_lines = self.test[self.test['user_id'] == user]
            rated_by_user = set(rated_by_user_lines.book_id.values)
            user_score = 0
            for i in range(k):
                if books_id_recommended[i] in rated_by_user:
                    user_score += 1/(i+1)
            users_precision += user_score
        return users_precision / len(group_by_user.index)

    def RMSE(self):
        pred_table, _ = self.user_based.build_CF_prediction_matrix('cosine')
        _, uniques = pd.factorize(self.ratings.book_id)
        book_id_dict = dict()
        for i in range(len(uniques.values)):
            book_id_dict[uniques.values[i]] = i
        compare = 0
        for line in self.test.values:
            user_id = line[0] - 1
            book_id = line[1]
            rating = line[2]

            normalized_book_id = book_id_dict[book_id]
            compare += (rating - pred_table[user_id][normalized_book_id]) ** 2
        rmse = math.sqrt(compare / len(self.test.values))
        return rmse

    def pre_possessor(self, k):
        current_test = self.test.copy()
        current_test = current_test[current_test['rating'] >= 4]
        group_by_user = current_test.groupby('user_id').size()
        # drop all user that voted least then k books
        current_test = current_test.copy()
        for user, count in zip(group_by_user.index, group_by_user.values):

            if count < k:
                current_test = current_test.drop(current_test[current_test.user_id == user].index)

        return current_test
