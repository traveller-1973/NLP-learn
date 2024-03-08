import torch
import torch.nn as nn
from model import transformer_embedding, Multi_head_attention, PositionwiseFeedForward, LayerNorm


class encodeer_layer(nn.Module):
    def __init__(self, d_model, n_head ,drop_prob):
        super(encodeer_layer, self).__init__()
        self.multi_head_attention = Multi_head_attention(d_model, n_head)
        self.layer_norm1 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, 512)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=drop_prob)


    def forward(self, x, src_mask):
        # x: [batch_size,seq_len,d_model]
        # return: [batch_size,seq_len,d_model]
        _x = x
        x = self.multi_head_attention(q=x, k=x, v=x, mask=src_mask)

        x = self.dropout(x)
        x = self.layer_norm1(x + _x)

        _x = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + _x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers, drop_prob,vocab_size,device):
        super(Encoder, self).__init__()
        self.embedding = transformer_embedding(d_model, vocab_size, 1000, device)
        self.layers = nn.ModuleList([encodeer_layer(d_model, n_head, drop_prob) for _ in range(num_layers)])

    def forward(self, x, src_mask):
        # x: [batch_size,seq_len]
        # src_mask: [batch_size,seq_len]
        # return: [batch_size,seq_len,d_model]
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = Multi_head_attention(d_model, n_head)
        self.layer_norm1 = LayerNorm(d_model)
        self.multi_head_attention = Multi_head_attention(d_model, n_head)
        self.layer_norm2 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, 512)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, enc_out, trg_mask, src_mask):
        # x: [batch_size,seq_len,d_model]
        # enc_out: [batch_size,seq_len,d_model]
        # return: [batch_size,seq_len,d_model]
        _x = x
        x = self.masked_multi_head_attention(q=x, k=x, v=x, mask=trg_mask)
        x = self.dropout(x)
        x = self.layer_norm1(x + _x)

        _x = x
        x = self.multi_head_attention(q=x, k=enc_out, v=enc_out, mask=src_mask)
        x = self.dropout(x)
        x = self.layer_norm2(x + _x)

        _x = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm3(x + _x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers, drop_prob, vocab_size, device):
        super(Decoder, self).__init__()
        self.embedding = transformer_embedding(d_model, vocab_size, 1000, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, drop_prob) for _ in range(num_layers)])

    def forward(self, x, enc_out, trg_mask, src_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc_out, trg_mask, src_mask)
        return x
    

class transformer(nn.Module):
    def __init__(self, d_model, n_head, num_layers, drop_prob, vocab_size, device):
        super(transformer, self).__init__()
        self.device = device
        self.encoder = Encoder(d_model, n_head, num_layers, drop_prob, vocab_size, device)
        self.decoder = Decoder(d_model, n_head, num_layers, drop_prob, vocab_size, device)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        # src, trg: [batch_size,seq_len]
        # src_mask, trg_mask: [batch_size,seq_len]

        src=src.to(self.device)
        trg=trg.to(self.device)
        src_mask=src_mask.to(self.device)
        trg_mask=trg_mask.to(self.device)
        src_mask = self.make_src_mask(src_mask)
        trg_mask = self.make_trg_mask(trg, trg_mask)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, enc_out, trg_mask, src_mask)
        out = self.linear(dec_out)
        return out
    
    def make_src_mask(self, src_mask):
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg, trg_mask):
        trg_pad_mask  = trg_mask.unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask