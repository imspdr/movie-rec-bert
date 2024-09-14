import pandas as pd
import json

movie_mapper = pd.read_csv("links.csv")

with open("../bert/item2index.json", "r") as file:
    item_map = json.load(file)

movie_id2tmdb_id = {}
for item_id in item_map.keys():
    find = movie_mapper[movie_mapper["movieId"] == int(item_id)].index
    value = movie_mapper["tmdbId"].loc[find]
    try:
        movie_id2tmdb_id[item_id] = str(int(value))
    except ValueError:
        continue
with open("movie_id2tmdb_id.json", "w") as file:
    json.dump(movie_id2tmdb_id, file)

print(len(movie_id2tmdb_id))
print(len(item_map))
