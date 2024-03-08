import time
from sklearn import logger
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



with open('天龙八部.txt', 'r',encoding='utf-8') as f:
    text = f.read()
chars=sorted(list(set(text)))



batch_size=32
context_length=512
d_model=512
n_head=8
vocab_size=len(chars)
learning_rate=3e-4
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



char_to_idx={ch:i for i,ch in enumerate(chars)}
idx_to_char={i:ch for i,ch in enumerate(chars)}
def encode(text):
    return [char_to_idx[ch] for ch in text]
def decode(text):
    return ''.join([idx_to_char[i] for i in text])


data=torch.tensor(encode(text),dtype=torch.long)
n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]

class train_dataset(Dataset):
    def __init__(self, data, context_length):
        super(train_dataset, self).__init__()
        self.data=data
        self.context_length=context_length

    def __len__(self):
        return len(self.data)-self.context_length-1
    
    def __getitem__(self,i):
        # 错开一位
        return (self.data[i:i+self.context_length].to(device),self.data[i+1:i+self.context_length+1].to(device))
    
train_set=train_dataset(train_data,context_length)
train_dataloader=DataLoader(train_set,batch_size=batch_size,shuffle=True)


class embedding(nn.Module):
    def __init__(self, vocab_size,context_length,d_model):
        super(embedding, self).__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.positonal_embedding=nn.Embedding(context_length,d_model)
        # self.lm_head=nn.Linear(d_model,vocab_size)

    def forward(self,src):
        # src,trg: (batch_size, context_length)
        # return: (batch_size, context_length, d_model)

        batch_size,context_length=src.shape

        word_embedidng=self.embedding(src) #(batch_size, context_length, d_model)
        pos_embedding=self.positonal_embedding(torch.arange(context_length).to(device)) #(context_length, d_model)
        embedding=word_embedidng+pos_embedding #(batch_size, context_length, d_model)

        return embedding
    
    
class layer_norm(nn.Module):
    def __init__(self, d_model):
        super(layer_norm, self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (batch_size, context_length, d_model)
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,keepdim=True)
        out=self.gamma*((x-mean)/(var+1e-12))+self.beta
        # out: (batch_size, context_length, d_model)

        return out

class head(nn.Module):
    def __init__(self, d_model, d_head, context_length):
        super(head, self).__init__()
        self.q=nn.Linear(d_model,d_head)
        self.k=nn.Linear(d_model,d_head)
        self.v=nn.Linear(d_model,d_head)
        # self.out=nn.Linear(d_model,d_model)

        # 下三角矩阵
        self.register_buffer('mask',torch.tril(torch.ones(context_length,context_length)))

    def forward(self, x):
        # x: (batch_size, context_length, d_model)
        
        batch_size,context_length,d_model=x.shape

        # q,k,v: (batch_size, context_length, d_head)
        q=self.q(x)
        k=self.k(x)
        v=self.v(x)

        score=q @ k.transpose(-2,-1) *d_model**-0.5
        score=score.masked_fill(self.mask[:context_length,:context_length]==0,float('-inf'))
        score=F.softmax(score,dim=-1)
        # score: (batch_size, context_length, context_length)

        out=score @ v
        # out: (batch_size, context_length, d_head)

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, context_length):
        super(MultiHeadAttention, self).__init__()

        # 创建n_head个head，其中d_head=d_model//n_head
        self.heads=nn.ModuleList([head(d_model,d_model//n_head,context_length) for _ in range(n_head)])
        self.out=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch_size, context_length, d_model)
        batch_size,context_length,d_model=x.shape

        out=torch.cat([head(x) for head in self.heads],dim=-1)
        # out: (batch_size, context_length, d_model)
        out=self.out(out)
        self.dropout=nn.Dropout(0.1)
        # out: (batch_size, context_length, d_model)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.fc1=nn.Linear(d_model,d_model*4)
        self.fc2=nn.Linear(d_model*4,d_model)
        self.dropout=nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch_size, context_length, d_model)
        out=F.relu(self.fc1(x))
        out=self.fc2(out)
        self.dropout=nn.Dropout(0.1)
        # out: (batch_size, context_length, d_model)

        return out   

class block(nn.Module):
    def __init__(self, d_model, n_head, context_length):
        super(block, self).__init__()
        self.self_attention=MultiHeadAttention(d_model,n_head,context_length)
        self.feed_forward=FeedForward(d_model)
        self.ln1=layer_norm(d_model)
        self.ln2=layer_norm(d_model)

    def forward(self, x):
        # x: (batch_size, context_length, d_model)
        # 先进行layermorm，然后进行self_attention后残差连接
        out=x+self.self_attention(self.ln1(x))
        out=x+self.feed_forward(self.ln2(out))
        # out: (batch_size, context_length, d_model)

        return out
    
class gpt(nn.Module):
    def __init__(self, vocab_size,context_length,d_model):
        super(gpt,self).__init__()

        self.context_length=context_length
        self.embedding=embedding(vocab_size,context_length,d_model)
        self.blocks=nn.ModuleList([block(d_model,n_head,context_length) for _ in range(6)])
        self.lm_head=nn.Linear(d_model,vocab_size)

    def forward(self,src,trg=None):
        # src,trg: (batch_size, context_length)
        embedding=self.embedding(src)
        # embedding: (batch_size, context_length, d_model)

        out=embedding
        for block in self.blocks:
            out=block(out)
        # out: (batch_size, context_length, d_model)
            
        logits=self.lm_head(out)
        # logits: (batch_size, context_length, vocab_size)
        
        if trg is None:
            loss=None
        else:
            logits=logits.view(logits.shape[0]*logits.shape[1],-1)
            target=trg.view(-1)
            loss=F.cross_entropy(logits,target)
        return logits,loss
    
    def generate(self, idx, max_tokens):
        # idx: (batch_size, context_length)
        for _ in range(max_tokens):
            # positonal_embedding有最大长度context_length，需要将idx的长度截断
            idx_cond=idx[:,-self.context_length:]

            logits,loss=self(idx_cond)

            # 最后一个字符的logits包含上下文
            logits=logits[:,-1,:]
            prob=F.softmax(logits,dim=1)
            idx_next=torch.multinomial(prob,1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx
    

model=gpt(vocab_size,context_length,d_model).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

print(len(train_dataloader))

for epoch in range(10):
    logger.info('start time:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    for step,(src,trg) in enumerate(train_dataloader):
        src=src.to(device)
        trg=trg.to(device)
        logits,loss=model(src,trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%1000==0:
            print(loss.item())
        # print(loss.item())
        logger.info('end time:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    
    with open ('generate{}.txt'.format(epoch+1),'w',encoding='utf-8') as f:
        idx =torch.zeros((1,1),dtype=torch.long).to(device)
        out=decode(model.generate(idx,10000)[0].tolist())
        f.write(out)