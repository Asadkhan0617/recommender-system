from recommender import *
from evaluate import precision_at_k
from evaluate import precision_at_k, recall_at_k, f1_score_at_k

# ---------- LOAD DATA ----------
data, movies = prepare_data("data/ratings.csv", "data/movies.csv")

# ---------- COLLABORATIVE ----------
user_item_matrix = create_user_item_matrix(data)
user_similarity = compute_user_similarity(user_item_matrix)

# ---------- CONTENT ----------
content_similarity = compute_content_similarity(movies)

# ---------- INPUT ----------
user_id = int(input("Enter user ID (e.g., 1): "))
movie_name = input("Enter a movie you like: ")

# ---------- COLD START ----------
if user_id not in user_item_matrix.index:
    print("\n⚠️ New user detected → showing popular movies\n")

    popular = data.groupby('title')['rating'].mean().sort_values(ascending=False)

    for movie in popular.head(10).index:
        print(movie)

    exit()

# ---------- GET RECOMMENDATIONS ----------
collab_recs = recommend_collaborative(user_id, user_item_matrix, user_similarity)
content_recs = recommend_content(movie_name, movies, content_similarity)

if len(collab_recs) == 0 and len(content_recs) == 0:
    print("❌ No recommendations found")
    exit()

# ---------- HYBRID ----------
final_recs = hybrid_recommendation(collab_recs, content_recs)

# Remove duplicates
final_recs = list(dict.fromkeys(final_recs))

print("\n🎬 Recommended Movies:")
for movie in final_recs:
    print(movie)

# ---------- EVALUATION ----------
user_history = data[data['userId'] == user_id]

# Only consider liked movies
relevant_items = user_history[user_history['rating'] >= 4]['title'].tolist()

precision = precision_at_k(final_recs, relevant_items)
recall = recall_at_k(final_recs, relevant_items)
f1 = f1_score_at_k(precision, recall)

print(f"\nPrecision@10: {precision:.2f}")
print(f"Recall@10: {recall:.2f}")
print(f"F1 Score@10: {f1:.2f}")
