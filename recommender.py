import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------- DATA PREP ----------
def prepare_data(ratings_path, movies_path):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    data = pd.merge(ratings, movies, on="movieId")
    return data, movies


# ---------- COLLABORATIVE ----------
def create_user_item_matrix(data):
    matrix = data.pivot_table(index='userId', columns='title', values='rating')
    return matrix.fillna(0)


def compute_user_similarity(user_item_matrix):
    return cosine_similarity(user_item_matrix)


def recommend_collaborative(user_id, user_item_matrix, similarity, top_n=10):
    if user_id not in user_item_matrix.index:
        return pd.Series()

    user_index = user_item_matrix.index.get_loc(user_id)
    sim_scores = similarity[user_index]

    similar_users = np.argsort(sim_scores)[::-1][1:6]

    recommendations = user_item_matrix.iloc[similar_users].mean()
    recommendations = recommendations.sort_values(ascending=False)

    already_watched = user_item_matrix.iloc[user_index]
    recommendations = recommendations[already_watched == 0]

    return recommendations.head(top_n)


# ---------- CONTENT-BASED ----------
def compute_content_similarity(movies):
    tfidf = TfidfVectorizer()

    movies['content'] = movies['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['content'])

    return cosine_similarity(tfidf_matrix)


def recommend_content(movie_title, movies, similarity, top_n=10):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    if movie_title not in indices:
        print("❌ Movie not found in dataset")
        return []

    idx = indices[movie_title]
    sim_scores = list(enumerate(similarity[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]


# ---------- NORMALIZATION ----------
def normalize(series):
    if len(series) == 0:
        return series
    if series.max() == series.min():
        return series
    return (series - series.min()) / (series.max() - series.min())


# ---------- HYBRID ----------
def hybrid_recommendation(user_recs, content_recs, alpha=0.5):
    hybrid_scores = {}

    # Normalize collaborative scores
    user_recs = normalize(user_recs)

    # Add collaborative scores
    for movie in user_recs.index:
        hybrid_scores[movie] = alpha * user_recs[movie]

    # Add content-based scores (weighted by rank)
    for i, movie in enumerate(content_recs):
        score = (1 - alpha) * (len(content_recs) - i) / len(content_recs)

        if movie in hybrid_scores:
            hybrid_scores[movie] += score
        else:
            hybrid_scores[movie] = score

    sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    return [movie for movie, score in sorted_recs[:10]]