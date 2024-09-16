import logging
from typing import Dict
import numpy as np
import kserve
import json
import torch
import pandas as pd
import torch.utils.data as data_utils
from bert4rec.dataset import BertEvalDataset


def try_or_default(dict, key, default_value):
    try:
        ret = dict[key]
    except KeyError:
        ret = default_value
    return ret

def is_number(v):
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False

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

class MovieBert(kserve.Model):
    def __init__(self, name):
        super().__init__(name)
        self.ready = False
        self.model_name = name
        self.model = None
        self.item2index = None
        self.index2item = None

    def load(self) -> bool:
        logging.info("load : use bert model")
        self.model = torch.load("bert4rec_model")
        with open("index2item.json", "r") as file:
            self.index2item = json.load(file)
        with open("item2index.json", "r") as file:
            self.item2index = json.load(file)

        self.ready = True
        return self.ready

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        try:
            given_data = payload["instances"]
        except KeyError:
            return {
                "predictions": []
            }

        params = try_or_default(payload, "params", {})
        topk = try_or_default(params, "topk", 10)
        print(given_data)
        print(topk)
        try:
            index_data = []
            for item in list(given_data):
                if str(item) in self.item2index.keys():
                    index_data.append(self.item2index[str(item)])
            print(index_data)
            dataset = BertEvalDataset([index_data])
            dataloader = data_utils.DataLoader(dataset, batch_size=128,
                                               shuffle=True, pin_memory=True)

            self.model.eval()
            for batch in dataloader:
                batch = [x.to("cpu") for x in batch]
                print(batch)
                result = self.model(batch[0])
                result = result[:, -1, :]
                scores = pd.DataFrame(result.cpu().detach().numpy().reshape(-1))
                scores = scores.sort_values(by=0, ascending=False, ignore_index=False)
                top_k_items = scores[:topk].index.to_list()
            print(top_k_items)
            predictions = []
            for item in top_k_items:
                predictions.append(self.index2item[str(item)])

            result = {
                "predictions": predictions,
            }
            json_result = json.dumps(obj=result, cls=NpEncoder, indent=4, ensure_ascii=False)
            return json.loads(json_result)

        except Exception as e:
            raise Exception("Failed to predict %s" % e)

if __name__ == "__main__":
    model = MovieBert("movie-rec-bert-serving")
    kserve.ModelServer().start([model])