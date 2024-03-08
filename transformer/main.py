import os
import sys
from sklearn import logger
import torch
import torch.nn as nn
import math
from dataloader import TrainDataset
from transformers import BertTokenizer, BertModel
from transformer import transformer
from nltk.translate.bleu_score import sentence_bleu

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.chdir(sys.path[0])



tokenizer_en = BertTokenizer.from_pretrained('D:\\OneDrive\\code\\model\\HuggingFace\\bert-base-uncased')
tokenizer_zh = BertTokenizer.from_pretrained('D:\\OneDrive\\code\\model\\HuggingFace\\bert-base-chinese')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

d_model = 128
n_head = 8
num_layers = 6
drop_prob = 0.1
batch_size = 8
vocab_size = max(tokenizer_en.vocab_size, tokenizer_zh.vocab_size)
epoch=10

model=transformer(d_model, n_head, num_layers, drop_prob, vocab_size, device).to(device)


train_dataset = TrainDataset('data/train.txt', tokenizer_en, tokenizer_zh)
valid_dataset = TrainDataset('data/val.txt', tokenizer_en, tokenizer_zh)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (trg, trg_mask, src, src_mask) in enumerate(dataloader):

        optimizer.zero_grad()
        output = model(src, trg[:, :-1],src_mask, trg_mask[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1).to(device)
        # output和trg错开一位，所以trg从第二位开始

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(dataloader)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, (trg, trg_mask, src, src_mask) in enumerate(dataloader):
            output = model(src, trg[:, :-1], src_mask, trg_mask[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            _trg=trg
            trg = trg[:, 1:].contiguous().view(-1).to(device)
            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()
        
            total_bleu = []
            for j in range(batch_size):
                trg_words = tokenizer_en.convert_ids_to_tokens(_trg[j])
                output_words = output[j].max(dim=1)[1]
                # output: [batch_size, seq_len, vocab_size]
                # output_words: [seq_len], 在dim=1上取最大值的索引

                output_words = tokenizer_en.convert_ids_to_tokens(output_words)
                bleu = sentence_bleu(hypothesis=output_words, references=trg_words)
                total_bleu.append(bleu)

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(dataloader), batch_bleu

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        train_loss = train(model, train_dataloader, optimizer, criterion, 1)
        valid_loss, bleu = evaluate(model, valid_dataloader, criterion)

        logger.info('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, BLEU: {:.4f}'.format(step + 1, total_epoch, train_loss, valid_loss, bleu))

        # if step > warmup:
        #     scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

run (epoch, 10000)