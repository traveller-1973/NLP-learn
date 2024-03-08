import torch
import transformers
import json
from transformers import BertTokenizer, BertModel


# with open('data\\train.txt', 'r', encoding='utf-8') as file_train:
#     data_train = file_train.readlines()
#     print(data_train[11])
#     data=json.loads(data_train[11])
#     print(tokenizer_en.encode(data[0]))
#     print(tokenizer_zh.tokenize(data[1]))


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, tokenizer_en, tokenizer_zh):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_zh = tokenizer_zh
        with open(file_name, 'r', encoding='utf-8') as file:
            self.data = file.readlines()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = json.loads(self.data[idx])
        encoded_en = self.tokenizer_en.encode_plus(
            data[0],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'  # 返回PyTorch张量
        )
        encoded_zh = self.tokenizer_zh.encode_plus(
            data[1],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt' 
        )
        return encoded_en['input_ids'].squeeze(0),encoded_en['attention_mask'].squeeze(0),encoded_zh['input_ids'].squeeze(0),encoded_zh['attention_mask'].squeeze(0)


