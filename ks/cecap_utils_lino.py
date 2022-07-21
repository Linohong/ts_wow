from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def sort_candidates_by(method, dialogue_history, candidates):
    '''
        this module is currently (2022.01.28) necessary
        only at the inference state.
        including the sorting objective to the training phase
        will be considered as the future work.
    '''
    if method == 'tf-idf':
        # corpus = [
        #             'This is the first document.',
        #             'This document is the second document.',
        #             'And this is the third one.',
        #             'Is this the first document?',
        #         ]
        corpus = [dialogue_history] + candidates

        # Initialize an instance of tf-idf Vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Generate the tf-idf vectors for the corpus
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

        # compute and print the cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        sorted_indices = [i[0] for i in sorted(enumerate(cosine_sim[0,:]), key=lambda k: k[1], reverse=True) if i[0] != 0] # index 0 is a dialogue history itself.
        sorted_candidates = [candidates[i-1] for i in sorted_indices]  # exclude 0, thus -1.

    return sorted_candidates