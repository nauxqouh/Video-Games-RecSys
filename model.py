from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

DB_URL = "sqlite:///games.db"

def get_engine():
    return create_engine(DB_URL)

def get_data():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM games", engine)

def get_all_game_names():
    return get_data()["name"].tolist()

tf = joblib.load("tfidf_vectorizer.joblib")
svd = joblib.load("svd.joblib")
svd_matrix = joblib.load("svd_matrix.joblib")
nn = joblib.load("nn.joblib")

def svd_game_recommendations(game_1, game_2, game_3, game_4, game_5, algorithm='cosine'):
    # Recording elapsed time
    start = time.time()
    
    # Dataframe
    df = get_data()

    # Input IDS
    ## Checks for the datatype of the inputted games either None or the title of the game
    input_ids = []
    for x in [game_1, game_2, game_3, game_4, game_5]:
        if x is not None:
            if x in df['name'].values:
                input_ids.append(df[df['name'] == x].index[0])
            else:
                return f"Game '{x}' is not in the dataset."

    # Iterate through each game selected and append the game's description into a list  
    game_text_list = []
    for x in [game_1, game_2, game_3, game_4, game_5]:
        if x is not None and x in df['name'].values:
            game_text_list.append(df[df['name'] == x]['total_contents'].values[0])
    
    # Concatenate the strings
    game_text_strings = ''
    for x in game_text_list:
        game_text_strings += x 
    
    # Apply TF-IDF and TruncatedSVD to user_input
    user_tf_idf = tf.transform([game_text_strings])
    user_svd = svd.transform(user_tf_idf.toarray())
    
    #Similarity/distance matrix
    if algorithm == 'cosine':
        # user_vector = svd_matrix[idx_user]
        # sim_scores = cosine_sim_row(user_vector, pd.DataFrame(svd_matrix[:-1]))
        similarity = cosine_similarity(user_svd, svd_matrix).flatten()
        sim_scores = list(enumerate(similarity))
        sim_scores = [score for score in sim_scores if score[0] not in input_ids]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        game_indices = [i[0] for i in sim_scores[:10]]
        
    elif algorithm == 'knn':
        # Transforming the predictions
        distances, indices = nn.kneighbors(user_svd)
        game_indices = [i for i in indices[0] if i not in input_ids][:10]
    else:
        return 'invalid algorithm.'
    
    end = time.time()
    elapsed_time = end - start
    
    result = df.loc[game_indices, ['name', 'genre', 'image_url']].reset_index(drop=True)
    return result, elapsed_time
