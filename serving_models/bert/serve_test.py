import json
import torch
import pandas as pd
import torch.utils.data as data_utils
from bert4rec.dataset import BertEvalDataset

model = torch.load("bert4rec_model")
model.eval()

dataset = BertEvalDataset([[5,6,7]])
dataloader = data_utils.DataLoader(dataset, batch_size=128,
                                   shuffle=True, pin_memory=True)

for batch in dataloader:
    batch = [x.to("cpu") for x in batch]
    print(batch)
    result = model(batch[0])
    result = result[:, -1, :]
    scores = pd.DataFrame(result.cpu().detach().numpy().reshape(-1))
    scores = scores.sort_values(by=0, ascending=False, ignore_index=False)[0:10]

    top_k_items = scores.index.to_list()

with open("index2item.json", "r") as file:
    reverse_item_map = json.load(file)

last_result = []
for item in top_k_items:
    last_result.append(reverse_item_map[str(item)])
print(last_result)