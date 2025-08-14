import streamlit as st
from model import get_all_game_names, svd_game_recommendations
import joblib

st.set_page_config(page_title="🎮 Game Recommender", layout="wide")
st.title("🎮 Game Recommender System")

st.markdown("### 🔍 Select games that you like:")
games = get_all_game_names()
# game_1 = st.selectbox("Game 1", [""] + games)
# game_2 = st.selectbox("Game 2", [""] + games)
# game_3 = st.selectbox("Game 3", [""] + games)
# game_4 = st.selectbox("Game 4", [""] + games)
# game_5 = st.selectbox("Game 5", [""] + games)
selected_games = st.multiselect("Choose up to 5 games", games, max_selections=5)

game_1 = selected_games[0] if len(selected_games) > 0 else None
game_2 = selected_games[1] if len(selected_games) > 1 else None
game_3 = selected_games[2] if len(selected_games) > 2 else None
game_4 = selected_games[3] if len(selected_games) > 3 else None
game_5 = selected_games[4] if len(selected_games) > 4 else None

st.markdown("### 🔍 Select recommendation algorithm that you like:")
algorithm = st.selectbox("Choose recommendation algorithm", ["cosine", "knn"])

if st.button("🎯 Recommend"):
    with st.spinner("Loading..."):
        results = svd_game_recommendations(
            game_1 or None,
            game_2 or None,
            game_3 or None,
            game_4 or None,
            game_5 or None,
            algorithm=algorithm
        )
        if isinstance(results, str):
            st.error(results)
        else:
            df_results, elapsed_time = results
            st.success(f"⏱ Time Elapsed: {elapsed_time:.2f} seconds")
            st.markdown("### 🔝 Recommended Games:")
            cols = st.columns(5)
            for idx, row in df_results.iterrows():
                with cols[idx % 5]:
                    st.image(row["image_url"], use_container_width=True)
                    st.caption(f"**{row['name']}**\n\n*{row['genre']}*")