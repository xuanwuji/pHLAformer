#!/usr/bin/env python

# @Time    : 2024/3/28 11:48
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : pHLAformer.py

import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import math
from transformers import EsmTokenizer

# model config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = EsmTokenizer.from_pretrained('./ESM2_3B')
onehot_h = 32
d_model = 512
d_ff= 1024
pep_max_length = 14
hla_max_length = 372
d_k = d_v = 64
n_heads = 8
n_layers = 1

def seq_embedding(seqs_toks,embedding='one-hot'):
    if embedding=='one-hot':
        tok_embedding_file = './one-hot.csv'
    else:
        tok_embedding_file = './ESM2_3B/token_embedding.csv'
    df = pd.read_csv(tok_embedding_file)
    seqs_embs = []
    for seq_tok in seqs_toks['input_ids']:
        seq_emb = []
        for tok in seq_tok:
            tok_emb = df.iloc[int(tok)].values.tolist()
            seq_emb.append(tok_emb)
        seqs_embs.append(seq_emb)
    return torch.tensor(seqs_embs).to(device)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(1) is PAD token
    k_pad_attn_mask = seq_k.data.eq(1).unsqueeze(1)
    k_pad_attn_mask = k_pad_attn_mask.expand(batch_size, len_q, len_k)
    q_pad_attn_mask = seq_q.data.eq(1).unsqueeze(-1)
    q_pad_attn_mask= q_pad_attn_mask.expand(batch_size, len_q, len_k)
    pad_attn_mask = k_pad_attn_mask | q_pad_attn_mask
    return pad_attn_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SEQembedding(nn.Module):
    def __init__(self,max_length):
        super(SEQembedding,self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.l1 = nn.Linear(onehot_h,d_model)
        self.act = nn.GELU()
        self.max_length = max_length
    def forward(self,seqs):
        seqs_toks = tokenizer(seqs, return_tensors='pt', add_special_tokens=False, padding='max_length', max_length=self.max_length)
        seqs_embs = seq_embedding(seqs_toks)
        seqs_embs = self.l1(seqs_embs)
        seqs_embs = self.act(seqs_embs)
        seqs_embs = self.pos_emb(seqs_embs.transpose(0,1)).transpose(0,1)
        return seqs_embs, seqs_toks['input_ids']

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask.to(device), -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn  = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = nn.GELU()(self.conv1(inputs.transpose(1, 2)))
        output = nn.GELU()(self.conv2(output).transpose(1, 2))
        return self.layer_norm(output + residual)


class CrossAttentionLayer(nn.Module):
    def __init__(self):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, query_emb, key_emb, attn_mask):
        outputs, attn = self.cross_attn(Q=query_emb, K=key_emb, V=key_emb, attn_mask=attn_mask)
        outputs = self.pos_ffn(outputs)
        return outputs, attn

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.PEPemb = SEQembedding(max_length=pep_max_length)
        self.HLAemb = SEQembedding(max_length=hla_max_length)
        self.layers = nn.ModuleList([CrossAttentionLayer() for _ in range(n_layers)])

    def forward(self, q_seq, k_seq,flag):
        if flag == 0:
            q_emb, q_tok = self.PEPemb(q_seq)
            k_emb, k_tok = self.HLAemb(k_seq)
            attn_mask = get_attn_pad_mask(q_tok,k_tok)
            cross_attns = []
            for layer in self.layers:
                q_emb, cross_attn = layer(q_emb, k_emb, attn_mask)
                cross_attns.append(cross_attn)
        if flag == 1:
            q_emb, q_tok = self.HLAemb(q_seq)
            k_emb, k_tok = self.PEPemb(k_seq)
            attn_mask = get_attn_pad_mask(q_tok, k_tok)
            cross_attns = []
            for layer in self.layers:
                q_emb, cross_attn = layer(q_emb, k_emb, attn_mask)
                cross_attns.append(cross_attn)
        return q_emb, cross_attns

class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear((pep_max_length+hla_max_length) * d_model, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 1))
    def forward(self,pep_feats,hla_feats):
        concat_feats = torch.cat((pep_feats.view(pep_feats.size(0),-1),hla_feats.view(hla_feats.size(0), -1)),dim=1)
        score = self.projection(concat_feats)
        return score

class pHLAformer(nn.Module):
    def __init__(self):
        super(pHLAformer, self).__init__()
        self.CrossAttention = CrossAttention().to(device)
        self.Projection = Projection().to(device)

    def forward(self, pep_seq, hla_seq):
        pep_feats , cross_attns = self.CrossAttention(pep_seq,hla_seq,flag=0)
        hla_feats, cross_attns = self.CrossAttention(hla_seq,pep_seq,flag=1)
        score = self.Projection(pep_feats, hla_feats)

