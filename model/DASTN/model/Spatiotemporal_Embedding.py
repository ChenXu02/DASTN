import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)
import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.DFT import get_domaintF
import time
class STEmbedding(nn.Module):
    def __init__(self,p,es_dim,et_dim,num_node,f):
        super(STEmbedding, self).__init__()
        self.p=p
        self.f=f
        self.es_dim=es_dim
        self.et_dim=et_dim
        self.num_node=num_node
        self.mapping=nn.Linear(self.p,self.et_dim)
        self.mapping2 = nn.Linear(1, self.es_dim)
        self.bias_sin = nn.Parameter(torch.FloatTensor(self.p))
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.es_dim), requires_grad=True)
    def forward(self,time_index,domaint_F):
        Code=None
        for p_i in range(0,self.p):
            code_temp=torch.sin(2*torch.pi*domaint_F[p_i]*time_index+self.bias_sin[p_i])
            if Code==None:
                Code=code_temp.unsqueeze(-1)
            else:
                Code=torch.cat((Code,code_temp.unsqueeze(-1)),1)
        TE=self.mapping2(self.mapping(Code).unsqueeze(-1)).permute(0,2,1)
        SE=self.node_embeddings
        STE= torch.matmul(SE, TE)
        return STE



