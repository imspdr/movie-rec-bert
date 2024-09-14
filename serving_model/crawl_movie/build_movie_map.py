import pandas as pd
import json
import requests
from bs4 import BeautifulSoup

movie_mapper = pd.read_csv("links.csv")

with open("../bert/item2index.json", "r") as file:
    item_map = json.load(file)

movie_id2tmdb_id = {}
movie_pool = []
for item_id in item_map.keys():
    find = movie_mapper[movie_mapper["movieId"] == int(item_id)].index
    value = str(int(movie_mapper["tmdbId"].loc[find]))
    movie_id2tmdb_id[item_id] = value
    url = f'https://www.themoviedb.org/movie/{value}?language=ko-KR'
    result = requests.get(url, headers={"User-agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(result, "html.parser")
    print(value)
    try:
        info = soup.find(class_="title ott_true").find_all("a")
    except AttributeError:
        try:
            info = soup.find(class_="title ott_false").find_all("a")
        except AttributeError:
            continue
    title = str(info[0].text)
    movie_pool.append({
        "id": int(item_id),
        "title": title
    })

with open("movie_id2tmdb_id.json", "w") as file:
    json.dump(movie_id2tmdb_id, file)

with open("../../frontend/src/store/movie_pool.json", "w", encoding="utf-8") as result_file:
    json.dump(movie_pool, result_file, ensure_ascii=False)

print(len(movie_id2tmdb_id))
print(len(item_map))
print(len(movie_pool))

