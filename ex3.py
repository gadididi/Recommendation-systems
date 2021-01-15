import pandas as pd

from Collaborativefiltering import CollaborativeFiltering
from ContactBasedFiltering import ContactBasedFiltering
from Non_personalized import NonPersonalizedRecommendation
from PrecisionMeasurement import PrecisionMeasurement


def main():
    books = pd.read_csv('books.csv', low_memory=False, encoding="ISO-8859-1")
    ratings = pd.read_csv('ratings.csv', low_memory=False)
    tags = pd.read_csv('tags.csv', low_memory=False)
    books_tags = pd.read_csv('books_tags.csv', low_memory=False)
    users = pd.read_csv('users.csv', low_memory=False)
    test = pd.read_csv('test.csv', low_memory=False)

    collab = CollaborativeFiltering(ratings, books, users)
    # collab.build_CF_prediction_matrix('cosine')
    collab.build_CF_prediction_matrix('euclidean')
    # collab.build_CF_prediction_matrix('jaccard')

    precision_measurement = PrecisionMeasurement(test, ratings, books, collab)

    # print(precision_measurement.precision_k(10))
    print(precision_measurement.ARHR(10))
    # print(precision_measurement.RMSE())


    # ## part one
    # non_personalized = NonPersonalizedRecommendation(books, ratings, users)
    # non_personalized.get_simply_recommendation(10)
    # non_personalized.get_simply_place_recommendation("Ohio", 10)
    # non_personalized.get_simply_age_recommendation(28,10)
    #
    # ## part two
    #
    # colab = CollaborativeFiltering(ratings, books)
    # colab.get_CF_recommendation(511, 10)
    #
    # ## part third
    # contact = ContactBasedFiltering(books)
    # contact.build_contact_sim_metrix()
    # contact.get_contact_recommendation("Twilight", 10)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
