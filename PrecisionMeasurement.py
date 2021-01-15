import math

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
        users_precision = 0
        pred_table, data_matrix = self.user_based.build_CF_prediction_matrix('cosine')
        rmse = 0
        users = 0
        none = 0
        for line in self.test.values:
            user_id = line[0]
            book_id = line[1]
            rating = line[2]
        #     books_calc = 0
        #     compare = 0
        #     for c in range(len(data_matrix[r])):
        #         if not math.isnan(data_matrix[r][c]):
        #             compare += (pred_table[r][c] - data_matrix[r][c]) ** 2
        #             books_calc += 1
        #     if books_calc != 0:
        #         rmse += math.sqrt(compare / books_calc)
        #         users += 1
        #     else:
        #         none += 1
        #
        # if users != 0:
        #     rmse = rmse/users

        print(none)
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
