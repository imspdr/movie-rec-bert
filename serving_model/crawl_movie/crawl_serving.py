from typing import Dict
import numpy as np
import kserve
import json
import requests
from bs4 import BeautifulSoup

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)

class MovieCrawl(kserve.Model):
    def __init__(self, name):
        super().__init__(name)
        self.ready = False
        self.model_name = name
        self.movie_id2tmdb_id = None
    def load(self) -> bool:
        with open("movie_id2tmdb_id.json", "r") as file:
            self.movie_id2tmdb_id = json.load(file)
        self.ready = True
        return self.ready

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        try:
            given_data = payload["instances"]
        except KeyError:
            return {
                "predictions": []
            }

        try:
            target = str(given_data[0])
            tmdb_id = self.movie_id2tmdb_id[target]
            url = f'https://www.themoviedb.org/movie/{tmdb_id}?language=ko-KR'

            result = requests.get(url, headers={"User-agent": "Mozilla/5.0"}).text
            soup = BeautifulSoup(result, "html.parser")
            img = str(soup.find(class_="poster w-full")["src"])
            try:
                info = soup.find(class_="title ott_true").find_all("a")
            except AttributeError:
                info = soup.find(class_="title ott_false").find_all("a")
            title = str(info[0].text)
            tags = list(map(lambda a: str(a.text), info[1:]))
            release = str(soup.find(class_="release").text).strip()
            description = str(soup.find(class_="overview").text).strip()

            predictions = [{
                "title": title,
                "tags": tags,
                "release": release,
                "description": description,
                "img": img
            }]

            result = {
                "predictions": predictions,
            }
            json_result = json.dumps(obj=result, cls=NpEncoder, indent=4, ensure_ascii=False)
            return json.loads(json_result)

        except Exception as e:
            raise Exception("Failed to predict %s" % e)

if __name__ == "__main__":
    model = MovieCrawl("movie-rec-bert-crawl")
    kserve.ModelServer().start([model])