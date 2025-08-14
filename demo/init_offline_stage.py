from algorithms_scratch.tfidf_fromscratch import *
from algorithms_scratch.truncatedsvd_fromscratch import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import joblib

df = pd.read_csv("video_games_dataset.csv")

# TF-IDF
tf = TfidfVectorizer(stop_words=stop_words_list, max_features=1250)
tf_idf_matrix = tf.fit_transform(df['total_contents'].values.astype('U'))

# SVD
tsv = TruncatedSVDFromScratch(n_components=401)
svd_matrix = tsv.fit_transform(tf_idf_matrix.toarray())

# KNN
nn_model = NearestNeighbors(n_neighbors=15, algorithm='ball_tree', metric='minkowski')
nn_model.fit(svd_matrix)

# Save
joblib.dump(tf, "tfidf_vectorizer.joblib")
joblib.dump(tf_idf_matrix, "tfidf_matrix.joblib")
joblib.dump(tsv, "svd.joblib")
joblib.dump(svd_matrix, "svd_matrix.joblib")
joblib.dump(nn_model, "nn.joblib")