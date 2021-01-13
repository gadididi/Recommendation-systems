from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class ContactBasedFiltering:
    def __init__(self, books):
        self.books = books
        self.similarity = None

    def build_contact_sim_metrix(self):
        books_sim_params = self.books[['book_id', 'original_title', 'authors']]
        # Apply clean_data function to your features.
        features = ['original_title', 'authors']
        for feature in features:
            books_sim_params[feature] = books_sim_params[feature].apply(clean_data)
        # Create a new soup feature
        books_sim_params['soup'] = books_sim_params.apply(create_soup, axis=1)
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(books_sim_params['soup'])
        # Compute the Cosine Similarity matrix based on the count_matrix
        self.similarity = cosine_similarity(count_matrix, count_matrix)

    def get_contact_recommendation(self, book_name, k):
        if self.similarity is None:
            self.build_contact_sim_metrix()

        indices = pd.Series(self.books.index, index=self.books['original_title'])

        idx = indices[book_name]
        if len(idx) > 1:
            idx = idx.values[0]
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.similarity[idx]))
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies (the first is the movie we asked)
        sim_scores = sim_scores[1:k + 1]
        recommended = []
        for inx, score in sim_scores:
            line = self.books[self.books["book_id"] == inx + 1]
            book = "book id: " + str(line.values[0][0]) + ", name: " + str(line.values[0][10])
            print(book)
            recommended.append(book)
        return recommended


def create_soup(x):
    return str(x['original_title']) + ' ' + str((x['authors']))


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
