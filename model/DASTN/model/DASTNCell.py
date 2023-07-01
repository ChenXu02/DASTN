import torch
import torch.nn as nn
from model.DAGNN import AVWGCN
from model.Spatiotemporal_Embedding import STEmbedding
from model.Real_time_correlation import MultiHeadAttention
import time
class DASTNCell(nn.Module):
    def __init__(self, p,f,es_dim,et_dim,Fr,node_num, dim_in, dim_out, cheb_k,embed_dim):
        super(DASTNCell, self).__init__()
        self.node_num = node_num
        self.p=p
        self.f=f
        self.es_dim=es_dim
        self.et_dim=et_dim
        self.Fr=Fr
        self.hidden_dim = dim_out
        self.cheb_k=cheb_k
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, self.cheb_k, self.et_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, self.cheb_k, self.et_dim)
        self.STE = STEmbedding(self.p, self.es_dim, self.et_dim, self.node_num,self.f)

    def forward(self, x, R,time_index,state,SC):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        STE = self.STE(time_index, self.Fr)
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, STE,R,SC))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, STE,R,SC))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)



class RGNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_node,num_heads,seq_len2):
        super(RGNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._seq_len2=seq_len2
        self.num_heads=num_heads
        self.num_node=num_node
        self.gate = nn.Linear( self._input_dim +self._seq_len2, self._input_dim)
        self.gate2 = nn.Linear(self.num_node, self._input_dim)
        self.atte=MultiHeadAttention(input_dim,hidden_dim,self.num_heads,self._seq_len2)


    def forward(self, inputs, hidden_state):
        hidden_state = hidden_state.to(inputs.device)
        inputs=inputs.squeeze(-1)
        #batch_size, seq_len2,num_nodes = inputs.shape
        #inputs = inputs.reshape((batch_size,num_nodes, seq_len2))
        #hidden_state = hidden_state.reshape((batch_size, num_nodes, self._input_dim))
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        g= torch.sigmoid(self.gate(concatenation))
        c = self.atte(concatenation)
        new_hidden_state = g * hidden_state + (1.0 - g) * torch.tanh(self.gate2(c))
        out=torch.relu(torch.tanh(c))
        return out,new_hidden_state
