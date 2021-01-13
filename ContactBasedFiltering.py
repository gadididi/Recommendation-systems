from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class ContactBasedFiltering:
    def __init__(self, books):
        self.books = books

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
        print(3)

    def get_contact_recommendation(self, book_name, k):
        pass


def create_soup(x):
    return ' '.join(x['original_title']) + ' ' + ' '.join(x['authors'])


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
