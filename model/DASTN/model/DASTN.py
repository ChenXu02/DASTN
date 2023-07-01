import torch
import torch.nn as nn
from model.DASTNCell import DASTNCell,RGNCell
import time

class AVWDCRNN(nn.Module):
    def __init__(self,args, Fr):
        super(AVWDCRNN, self).__init__()
        assert args.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self._hidden_dim_A = args.hidden_dim_A
        self._head_num_A = args.head_num_A
        self._input_dim_A = args.input_dim_A#dim of hidden state
        self.cheb_k=args.cheb_k
        self._seq_len2_A = args.lag2

        ##################################
        self.node_num = args.num_nodes
        self.input_dim = args.input_dim
        self.num_layers = args.num_layers
        self.end_convs = nn.ModuleList()
        self.GAU_cells = nn.ModuleList()
        self.GAU_cells.append(RGNCell(self._input_dim_A, self._hidden_dim_A,self.node_num, self._head_num_A, self._seq_len2_A))
        self.end_convs.append(nn.Conv2d(1, args.output_dim, kernel_size=(1, args.rnn_units), bias=True))

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(DASTNCell(args.p,args.f,args.embed_dim,args.embed_dim_t,Fr,args.num_nodes, args.input_dim, args.rnn_units,args.cheb_k, args.embed_dim))
        for _ in range(1, args.num_layers):
            self.end_convs.append(nn.Conv2d(1, args.output_dim, kernel_size=(1, args.rnn_units), bias=True))
            self.GAU_cells.append(RGNCell(self._input_dim_A, self._hidden_dim_A,self.node_num, self._head_num_A, self._input_dim_A))
            self.dcrnn_cells.append(DASTNCell(args.p,args.f,args.embed_dim,args.embed_dim_t,Fr,args.num_nodes, args.input_dim, args.rnn_units,args.cheb_k, args.embed_dim))

    def forward(self, x,data_local,time_index, init_state,init_state_A):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        current_inputs_A = data_local.permute(0,1,3,2,4)
        SC=torch.eye(self.node_num).unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)
        output_hidden = []
        output_hidden_A = []
        for i in range(self.num_layers):
            state = init_state[i]
            state_A = init_state_A[i]
            inner_states = []
            inner_states_A = []
            for t in range(seq_length):
                R,state_A = self.GAU_cells[i](current_inputs_A[:, t,:, :], state_A)
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :],R,time_index[:,t], state,SC)
                inner_states.append(self.end_convs[i](state.unsqueeze(1)).squeeze(1))
                inner_states_A.append(state_A)
            output_hidden.append(state)
            output_hidden_A.append(state_A)#state store
            current_inputs_pr = torch.stack(inner_states, dim=1)
            current_inputs_A = torch.stack(inner_states_A, dim=1)
            output=current_inputs_pr
            current_inputs=output+current_inputs
        return output, output_hidden, output_hidden_A

    def init_hidden(self, batch_size,num_nodes):
        init_states = []
        init_states_A = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
            init_states_A.append(torch.zeros(batch_size, num_nodes, self._input_dim_A))
        return torch.stack(init_states, dim=0),torch.stack(init_states_A, dim=0)      #(num_layers, B, N, hidden_dim)

class DASTN(nn.Module):
    def __init__(self, args,Fr):
        super(DASTN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.default_graph = args.default_graph
        self.encoder = AVWDCRNN(args,Fr)
        #predictor

    def forward(self, source,data_local, time_index,traing_assist,epoch_n):
        #source=torch.log(torch.nonzero(source))
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        lamb=0#min(1,epoch_n/70)
        data_local_save=data_local
        time_index_save=time_index
        seq_len2 = data_local.shape[2]
        state, state_A = self.encoder.init_hidden(source.shape[0], self.num_node)
        loop_c = 0
        output_multihorizon=None
        while (loop_c < self.horizon):
            output, state, state_A = self.encoder(source, data_local, time_index, state, state_A)  # B, T, N, hidden
            # CNN based predictor
            if output_multihorizon==None:
                output_multihorizon=output
            else:
                output_multihorizon=torch.cat((output_multihorizon,output),1)
            if traing_assist is not None and loop_c<self.horizon-1:
                output_recircle=output[:,-1,:,:].unsqueeze(1)#traing_assist[:,loop_c,:,:].unsqueeze(1)
            else:
                #print(output.shape,'sss')
                output_recircle=output[:,-1,:,:].unsqueeze(1)

            source = output_recircle#torch.cat((source, output), 1)[:, -seq_len:, :, :]
            data_local = torch.cat((data_local[:, -1, :, :, :], output_recircle), 1).unsqueeze(1)[:, :, -seq_len2:, :, :]
            time_index=(time_index[:,-1]+1).unsqueeze(0)
            #data_local=data_local_save[:, loop_c, :, :,:].unsqueeze(1)
            #time_index=time_index_save[:, loop_c].unsqueeze(1)
            loop_c+= 1
        return output_multihorizon

