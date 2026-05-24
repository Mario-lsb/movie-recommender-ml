import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


def item_similarity(ratings):

    matrix=ratings.pivot_table(
        index='movie',
        columns='user',
        values='rating'
    ).fillna(0)

    similarity=cosine_similarity(matrix)

    return similarity