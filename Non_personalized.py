def weighted_rating(v, R, m, C):
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


class NonPersonalizedRecommendation:
    def __init__(self, books, ratings, users):
        self.books = books
        self.ratings = ratings
        self.users = users

    def get_simply_recommendation(self, k):
        avg = self.ratings.groupby(['book_id']).mean()
        avg = avg.drop(['user_id'], axis=1)
        C = avg['rating'].mean()
        print(C)
        count_rate = self.ratings.pivot_table(index=['book_id'], aggfunc='size')
        m = count_rate.quantile(0.9)
        weighted_rating_table = avg.copy()
        weighted_rating_table['count_vote'] = count_rate
        q_movies = weighted_rating_table.copy().loc[weighted_rating_table['count_vote'] >= m]
        q_movies['score'] = weighted_rating(q_movies['count_vote'], q_movies['rating'], m, C)
        q_movies = q_movies.sort_values('score', ascending=False)
        q_movies["index"] = q_movies.index
        res = q_movies[['score']].head(k)
        l = res.index.tolist()
        l_fix = []
        for e in l:
            l_fix.append(e - 1)
        q_movies["book_name"] = self.books["original_title"].iloc[l_fix]

        print(res)
        return res

    def get_simply_place_recommendation(self, place, k):
        tmp = self.users.loc[self.users['location'] == place]
        tmp = tmp.join(self.ratings.set_index('user_id'), how='left', on='user_id')

        avg = tmp.groupby(['book_id']).mean()
        avg = avg.drop(['user_id'], axis=1)
        C = avg['rating'].mean()
        count_rate = tmp.pivot_table(index=['book_id'], aggfunc='size')
        m = count_rate.quantile(0.9)
        weighted_rating_table = avg.copy()
        weighted_rating_table['count_vote'] = count_rate
        q_movies = weighted_rating_table.copy().loc[weighted_rating_table['count_vote'] >= m]
        q_movies['score'] = weighted_rating(q_movies['count_vote'], q_movies['rating'], m, C)
        q_movies = q_movies.sort_values('score', ascending=False)
        q_movies["index"] = q_movies.index
        l = q_movies.index.tolist()
        l_fix = []
        for e in l:
            l_fix.append(e - 1)
        q_movies["book_name"] = self.books["original_title"].iloc[l_fix]
        res = q_movies[['book_name', 'score']].head(k)
        print(res)
        return res

    def get_simply_age_recommendation(self, age, k):
        low = (age % 10) * 10 + 1
        high = (age % 10) * 10 + 10
        tmp = self.users.loc[(self.users['age'] <= high) & (self.users['age'] >= low)]
        tmp = tmp.join(self.ratings.set_index('user_id'), how='left', on='user_id')
        avg = tmp.groupby(['book_id']).mean()
        avg = avg.drop(['user_id'], axis=1)
        C = avg['rating'].mean()
        count_rate = tmp.pivot_table(index=['book_id'], aggfunc='size')
        m = count_rate.quantile(0.9)
        weighted_rating_table = avg.copy()
        weighted_rating_table['count_vote'] = count_rate
        q_movies = weighted_rating_table.copy().loc[weighted_rating_table['count_vote'] >= m]
        q_movies['score'] = weighted_rating(q_movies['count_vote'], q_movies['rating'], m, C)
        q_movies = q_movies.sort_values('score', ascending=False)
        q_movies["index"] = q_movies.index
        l = q_movies.index.tolist()
        l_fix = []
        for e in l:
            l_fix.append(e - 1)
        q_movies["book_name"] = self.books["original_title"].iloc[l_fix]
        res = q_movies[['book_name', 'score']].head(k)
        print(res)
        return res
