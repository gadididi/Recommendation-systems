
from sklearn.feature_extraction.text import TfidfVectorizer


class ContactBasedFiltering:
    def __init__(self, books):
        self.books = books

    def build_contact_sim_metrix(self):
        books_sim_params = self.books[['book_id','original_title','authors']]
        books_sim_params['original_title'] = books_sim_params['original_title'].apply(clean_data)
        print(3)

    def get_contact_recommendation(self, book_name, k):
        pass

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''