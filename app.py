from flask import Flask, request, jsonify
from recommender import *

app = Flask(__name__)

# ---------- LOAD DATA ONCE ----------
data, movies = prepare_data("data/ratings.csv", "data/movies.csv")

user_item_matrix = create_user_item_matrix(data)
user_similarity = compute_user_similarity(user_item_matrix)
content_similarity = compute_content_similarity(movies)


# ---------- HOME ----------
@app.route("/")
def home():
    return "🎬 Recommendation System API is running"


# ---------- RECOMMENDATION ENDPOINT ----------
@app.route("/recommend", methods=["POST"])
def recommend():
    req_data = request.json

    user_id = req_data.get("user_id")
    movie_name = req_data.get("movie_name")

    # ---------- COLD START ----------
    if user_id not in user_item_matrix.index:
        popular = data.groupby('title')['rating'].mean().sort_values(ascending=False)
        return jsonify({
            "type": "cold_start",
            "recommendations": list(popular.head(10).index)
        })

    # ---------- GET RECOMMENDATIONS ----------
    collab_recs = recommend_collaborative(user_id, user_item_matrix, user_similarity)
    content_recs = recommend_content(movie_name, movies, content_similarity)

    final_recs = hybrid_recommendation(collab_recs, content_recs)

    # Remove duplicates
    final_recs = list(dict.fromkeys(final_recs))

    return jsonify({
        "type": "hybrid",
        "recommendations": final_recs
    })


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)