from bert4rec.model.bert import BERTModel
from bert4rec.trainer import Trainer
from bert4rec.dataset import BertTrainDataset
import pandas as pd
import torch.utils.data as data_utils
import json

# 파일 로드
file_path = "ratings.csv"
item_column = "movieId"
user_column = "userId"

df = pd.read_csv(file_path)

# 레이팅 낮은 거 안씀
min_rating = 2
df = df[df['rating'] >= min_rating]

# 조회 낮은 영화 삭제
item_refer_sizes = df.groupby(item_column).size()
good_items = item_refer_sizes.index[item_refer_sizes >= 3]
df = df[df[item_column].isin(good_items)]

# 조회 낮은 사용자 삭제
user_sizes = df.groupby(user_column).size()
good_users = user_sizes.index[user_sizes >= 2]
df = df[df[user_column].isin(good_users)]

num_users = df[user_column].nunique()

# item index 압축하고 mapper 기억
item_map = {s: i for i, s in enumerate(set(df[item_column]))}
reverse_item_map = {i: s for i, s in enumerate(set(df[item_column]))}

with open("item2index.json", "w") as file:
    json.dump(item_map, file, indent=4)
with open("index2item.json", "w") as file:
    json.dump(reverse_item_map, file, indent=4)

num_items = len(item_map)
user_group = df.groupby(user_column)

# train data 만들기
user2seq_data = {}
for i, user in enumerate(user_group):
    user2items = user[1].sort_values(by="timestamp")[item_column]
    user2seq_data[user[0]] = list(map(lambda d: item_map[d], list(user2items)))

dataset = BertTrainDataset(user2seq_data, num_items)
dataloader = data_utils.DataLoader(dataset, batch_size=128,
                                   shuffle=True, pin_memory=True)

# 모델 학습
model = BERTModel(num_items)

trainer = Trainer(model, dataloader)
trainer.train(200)

