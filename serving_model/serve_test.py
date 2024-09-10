import torch
from bert4rec.model.bert import BERTModel
from bert4rec.trainer import Trainer
from bert4rec.dataset import BertEvalDataset
import pandas as pd
import torch.utils.data as data_utils

model = torch.load("bert4rec_model")
model.eval()

seq = torch.LongTensor([1,2,3])
dataset = BertEvalDataset([[1,2,3]])
dataloader = data_utils.DataLoader(dataset, batch_size=128,
                                   shuffle=True, pin_memory=True)

for batch in dataloader:
    batch = [x.to("cpu") for x in batch]
    result = model(batch[0])
    result = result[:, -1, :]
    scores = pd.DataFrame(result.cpu().detach().numpy().reshape(-1))
    scores = scores.sort_values(by=0, ascending=False, ignore_index=False)[0:10]
    top_k_items = scores.index.to_list()

    print(top_k_items)