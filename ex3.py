import pandas as pd

from Non_personalized import NonPersonalizedRecommendation


def main():
    books = pd.read_csv('books.csv', low_memory=False, encoding="ISO-8859-1")
    ratings = pd.read_csv('ratings.csv', low_memory=False)
    tags = pd.read_csv('tags.csv', low_memory=False)
    books_tags = pd.read_csv('books_tags.csv', low_memory=False)
    users = pd.read_csv('users.csv', low_memory=False)
    test = pd.read_csv('test.csv', low_memory=False)

    non_personalized = NonPersonalizedRecommendation(books, ratings, users).get_simply_age_recommendation(55, 10)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
