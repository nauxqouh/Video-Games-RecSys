# Video-Games-RecSys

This repository contains the implementation and demo web app for Video Games Recommendation System, created as the final coursework for our Recommendation Systems course.

This project is adapted and further developed from a cloned GitHub repository [SulmanK - Video-Game-Recommendation-Engine](https://github.com/SulmanK/Video-Game-Recommendation-Engine) provided by the instructor as a baseline, with the requirement to understand how to build an end-to-end recommendation system and then extend it with our own implementations and experiments.

The goal of this project is to design a __content-based video game recommendation system__ using data from the __Giant Bomb API__. In addition to similarity-based methods, we also implemented __attribute filtering__ and __weighted scoring__ approaches to provide diverse recommendation strategies.

## Methods Implemented

1. __Content-Based Filtering__
   - __Vectorization__: TF-IDF applied to combined textual features (`genre`, `theme`, `concept`, `developer`, `franchise`, `platform`).
   - __Similarity Metrics__: Cosine Similarity & K-Nearest Neighbors (KNN).
   - __Dimensionality Reduction__: Truncated SVD to improve query speed.
   - __Offline stage optimization__: Precomputes heavy calculations to improve online query performance.

2. __Attribute Filtering__
   - __Single Attribute Filtering__: Suggests games matching individual attributes (e.g., same `genre` or `theme`).
   - __Combined Filtering__: Matches multiple attributes in priority order.
   - __Weighted Scoring__: Scores games based on weighted match of `genre`, `theme`, and `concept`.

## Performance Highlight

<div align="center">

| Method                      | Avg. Query Time | Notes                |
| --------------------------- | --------------- | -------------------- |
| Cosine Similarity (no SVD)  | 12–13s          | Accurate but slow    |
| KNN (no SVD)                | 7–8s            | Faster, more diverse |
| **SVD + Cosine Similarity** | 0.18–0.20s      | Accurate & fast      |
| **SVD + KNN**               | 0.05–0.07s      | Fastest & diverse    |

</div>

## Demo

The demo is implemented for the two fastest running algorithm above.

https://github.com/user-attachments/assets/43780887-137e-4bfb-b23c-25498d93727b

## Key Insights from the Coursework

- __Understanding of Content-Based Filtering__: Learned to build recommender systems without user data by leveraging only item attributes.
- __Algorithm Implementation Skills__: Implemented TF-IDF, cosine similarity, KNN, and Truncated SVD from scratch.
- __Integration of Offline & Online Stages__: Learned how to separate heavy computations into a preprocessing stage to deliver near real-time recommendations.
- __Optimization for Real-Time Use__: Saw the significant impact of dimensionality reduction and offline processing on query time.
- __Evaluation Beyond Accuracy__: Learned to assess recommendations via relevance and execution time in absence of traditional metrics.

## Repo Structure

```
Video-Games-RecSys/
│── algorithms_scratch/               # Core algorithm implementations (from scratch)
│   ├── cs_fromscratch.py             # Cosine Similarity implementation
│   ├── knn_fromscratch.py            # K-Nearest Neighbors implementation
│   ├── tfidf_fromscratch.py          # TF-IDF implementation
│   └── truncatedsvd_fromscratch.py   # Truncated SVD implementation
│
│── demo/                             # Web application demo of the recommendation system
│   ├── app.py                        # Main server application entry point
│   ├── init_db.py                    # Script to initialize the database
│   ├── init_offline_stage.py         # Script to prepare offline computations for recommendations
│   ├── model.py                      # Model-related functions and utilities
│   ├── requirement.txt               # Python dependencies for running the demo
│   └── video_games_dataset.csv       # Dataset used by the demo web application
│
│── README.md                         # Project documentation
│── fetched_video_games.csv           # Dataset used in the main project workflow
│── main.ipynb                        # Main Notebook demonstrating the complete end-to-end workflow
```

## Running the Demo app

#### 1️⃣ Install Dependencies

Navigate to the demo/ directory and install the required packages:
   
```
cd demo
pip install -r requirements.txt
```

#### 2️⃣ Initialize the Database

Set up the database for the recommendation system:

```
python init_db.py
```

#### 3️⃣ Prepare the Offline Stage

Run the offline initialization script to precompute required data:

```
python init_offline_stage.py
```

#### 4️⃣ Launch the Web App

Start the application server:

```
python app.py
```

By default, the app will be available at local server in your browser.

## Contributors

Nguyen Thi Xuan Huong, Chiem Huynh Giao, Nguyen Ngoc Bao Han.
