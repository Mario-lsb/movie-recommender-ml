import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


def user_similarity(ratings):

    matrix=ratings.pivot_table(
        index='user',
        columns='movie',
        values='rating'
    ).fillna(0)

    similarity=cosine_similarity(matrix)

    return similarity