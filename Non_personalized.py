import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class NonPersonalizedRecommendation:
    def __init__(self, books, ratings, users):
        self.books = books
        self.ratings = ratings
        self.users = users
        self.avg = self.ratings.groupby(['book_id']).mean()
        self.avg = self.avg.drop(['user_id'], axis=1)
        self.C = self.avg['rating'].mean()
        print(self.C)
        self.count_rate = self.ratings.pivot_table(index=['book_id'], aggfunc='size')
        self.m = self.count_rate.quantile(0.9)
        self.weighted_rating_table = self.avg.add(self.count_rate)
        print(self.weighted_rating_table)

    def get_simply_recommendation(self, k):
        q_movies = self.count_rate.copy().loc[self.count_rate >= self.m]

        self.weighted_rating(v,r)
        print(q_movies)

    def get_simply_place_recommendation(self, place, k):
        pass

    def get_simply_age_recommendation(self, age, k):
        pass

    # Function that computes the weighted rating of each movie
    def weighted_rating(self, v, R):
        # Calculation based on the IMDB formula
        return (v / (v + self.m) * R) + (self.m / (self.m + v) * self.C)
