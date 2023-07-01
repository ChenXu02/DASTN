import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads,seq_len2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.seq_len2=seq_len2


        self.W_query = nn.Linear(in_features=self.input_dim+self.seq_len2, out_features=self.input_dim*self.num_heads, bias=False)
        self.W_key = nn.Linear(in_features=self.input_dim+self.seq_len2, out_features=self.input_dim*self.num_heads, bias=False)
        self.W_h=nn.Linear(in_features=self.num_heads, out_features=1, bias=False)
    def forward(self, inputs):
        querys = self.W_query(inputs)  # [N, T_q, num_units]
        keys = self.W_key(inputs)  # [N, T_k, num_units]
        split_size = self.input_dim
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / ((self.hidden_dim+self.seq_len2) ** 0.5)
        scores = scores.permute(1,2,3,0)
        scores=self.W_h(scores)
        out = scores.squeeze(dim=-1)
        return out





