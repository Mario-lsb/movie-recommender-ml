from sklearn.decomposition import TruncatedSVD


def train_svd(ratings):

    matrix=ratings.pivot_table(
        index='user',
        columns='movie',
        values='rating'
    ).fillna(0)

    svd=TruncatedSVD(
        n_components=50
    )

    latent_matrix=svd.fit_transform(
        matrix
    )

    return svd,latent_matrix
