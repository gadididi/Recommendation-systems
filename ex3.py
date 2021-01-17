import pandas as pd

from Collaborativefiltering import CollaborativeFiltering
from ContactBasedFiltering import ContactBasedFiltering
from Non_personalized import NonPersonalizedRecommendation
from PrecisionMeasurement import PrecisionMeasurement


def main():
    """
    read all the relevant csv files. the files supposed to be in the same ex3.py directory
    :return: None
    """
    books = pd.read_csv('books.csv', low_memory=False, encoding="ISO-8859-1")
    ratings = pd.read_csv('ratings.csv', low_memory=False)
    tags = pd.read_csv('tags.csv', low_memory=False)
    books_tags = pd.read_csv('books_tags.csv', low_memory=False)
    users = pd.read_csv('users.csv', low_memory=False)
    test = pd.read_csv('test.csv', low_memory=False)

    """
    we designed the code with OOP. so you need to create first the specific class that you want to 
    test. there are 3 classes: 
    1) NonPersonalizedRecommendation 
        :parameters: books, ratings, users (all csv files)
    2) CollaborativeFiltering 
        :parameters: ratings, books, users (all csv files)
    3) ContactBasedFiltering 
        :parameter: books (csv file)
    
    create class you want to test and run the method that we were required in the exercise EX3
    
    ## there are another class- "PrecisionMeasurement". in this class there are all the accurate methods
    we  were required to implement (precision_k , ARHR , RMSE)
    """

    """
    part 1:
    NonPersonalizedRecommendation
    uncomment to run 
    """

    non_personalized = NonPersonalizedRecommendation(books, ratings, users)
    non_personalized.get_simply_recommendation(10)
    non_personalized.get_simply_place_recommendation("Ohio", 10)
    non_personalized.get_simply_age_recommendation(28,10)

    """
    part 2:
    CollaborativeFiltering
    uncomment to run 
    """

    # colab = CollaborativeFiltering(ratings, books, users)
    # colab.get_CF_recommendation(1, 10)

    """
    part 3:
    ContactBasedFiltering
    uncomment to run 
    """

    # contact = ContactBasedFiltering(books)
    # contact.build_contact_sim_metrix()
    # contact.get_contact_recommendation("Twilight", 10)

    """
    part 4:
    PrecisionMeasurement
    uncomment to run the accuracy tests : precision_k, ARHR, RMSE
    """
    # precision_measurement = PrecisionMeasurement(test, ratings, books, collab)

    # print(precision_measurement.precision_k(10))
    # print(precision_measurement.ARHR(10))
    # print(precision_measurement.RMSE())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
