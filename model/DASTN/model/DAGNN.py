import torch
import torch.nn.functional as F
import torch.nn as nn
import time
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out,cheb_k,  et_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k=cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(et_dim,self.cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(et_dim, dim_out))
    def forward(self, x, STE,R,SC):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        bias = torch.matmul(STE, self.bias_pool)  # N, dim_out
        R_temp=[SC,R]
        R_set =R_temp[0].unsqueeze(0)
        for k in range(2, self.cheb_k):
            R_temp.append(torch.matmul(2 * R, R_temp[-1]) - R_temp[-2])
        for i in range(1,len(R_temp)):
            R_set=torch.cat((R_set, R_temp[i].unsqueeze(0)), 0)
        R_set = R_set.permute(1, 2, 0, 3)
        x_gconv = torch.einsum('bnkm,bmi,bnd,dkio->bno', R_set, x, STE, self.weights_pool) + bias     #b, N, dim_out
        return x_gconv
