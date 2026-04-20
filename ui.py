import streamlit as st
import requests

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎬 Smart Movie Recommendation System")

st.write("Get personalized movie recommendations using AI 🚀")

# ---------- INPUT ----------
user_id = st.number_input("Enter User ID", min_value=1, value=1)
movie_name = st.text_input("Enter a movie you like", "Toy Story (1995)")

# ---------- BUTTON ----------
if st.button("Get Recommendations"):
    url = "http://127.0.0.1:5000/recommend"

    payload = {
        "user_id": user_id,
        "movie_name": movie_name
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()

            st.success(f"Model Used: {data['type']}")

            st.subheader("🎯 Recommended Movies:")
            for movie in data["recommendations"]:
                st.write(f"👉 {movie}")

        else:
            st.error("Error fetching recommendations")

    except:
        st.error("⚠️ Make sure Flask API is running!")