import pandas as pd
import sqlite3

# read file data .csv
data = pd.read_csv("video_games_dataset.csv")

# connect to file SQLite (auto create if not available)
conn = sqlite3.connect("games.db")

# update data into table named "games"
data.to_sql("games", conn, if_exists="replace", index=False)
print("Update data successfully.")

conn.close()