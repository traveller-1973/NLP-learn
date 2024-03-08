import transformers
from transformers import BertTokenizer, BertModel
import json

tokenizer_en = BertTokenizer.from_pretrained('D:\\OneDrive\\code\\model\\HuggingFace\\bert-base-uncased')
tokenizer_zh = BertTokenizer.from_pretrained('D:\\OneDrive\\code\\model\\HuggingFace\\bert-base-chinese')


print(tokenizer_en.vocab_size)
print(tokenizer_zh.vocab_size)
i=0
with open('data\\news-commentary-v13.zh-en.en', 'r', encoding='utf-8') as file_en, \
     open('data\\news-commentary-v13.zh-en.zh', 'r', encoding='utf-8') as file_zh:
    data_en = file_en.readlines()
    data_zh = file_zh.readlines()

# 训练集
with open("data\\train.txt", 'w', encoding='utf-8') as file_train:
    for i in range(202221):
        line_to_write = json.dumps([data_en[i].strip(), data_zh[i].strip()],ensure_ascii=False)+ '\n'
        file_train.write(line_to_write)
        # file_train.write(data_zh[i])

# 验证集
with open("data\\val.txt", 'w', encoding='utf-8') as file_val:
    for i in range(202221, 202221+25278):
        line_to_write = json.dumps([data_en[i].strip(), data_zh[i].strip()],ensure_ascii=False)+ '\n'
        file_val.write(line_to_write)
        # file_val.write(data_zh[i])

# 测试集
with open("data\\test.txt", 'w', encoding='utf-8') as file_test:
    for i in range(202221+25278, 202221+25278+25278):
        line_to_write = json.dumps([data_en[i].strip(), data_zh[i].strip()],ensure_ascii=False)+ '\n'
        file_test.write(line_to_write)
        # file_test.write(data_zh[i])




