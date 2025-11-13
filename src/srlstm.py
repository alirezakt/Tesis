'''
Main Models
Author: Pu Zhang
Date: 2019/7/1
'''
import torch
import torch.nn as nn
from .utils import *
from .basemolel import *

class SR_LSTM(nn.Module):
    def __init__(self, args):
        super(SR_LSTM, self).__init__()
        
        args.seq_length = 20
        args.obs_length = 8
        args.pred_length = 12
        self.args = args
        self.ifdropout = True
        self.using_cuda = True
        self.inputLayer = nn.Linear(2, 32)
        self.cell = LSTMCell(32, 64)

        self.gcn = GCN(args,32, 64)

        self.gcn1 = GCN(args, 32, 64)

        self.outputLayer = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.1)

        self.input_Ac = nn.ReLU()


        self = self.cuda()
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant(self.inputLayer.bias, 0.0)
        nn.init.normal(self.inputLayer.weight, std=0.2)

        nn.init.xavier_uniform(self.cell.weight_ih)
        nn.init.orthogonal(self.cell.weight_hh, gain=0.001)

        nn.init.constant(self.cell.bias_ih, 0.0)
        nn.init.constant(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant(self.outputLayer.bias, 0.0)
        nn.init.normal(self.outputLayer.weight, std=0.1)

    def forward(self, inputs,iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum=inputs
        
        nodes_abs = nodes_abs[:, :, :2]
        nodes_norm = nodes_norm[:, :, :2]
        shift_value = shift_value[:, :, :2]

        
        num_Ped = nodes_norm.shape[1]
        
        outputs=torch.zeros(nodes_norm.shape[0],num_Ped, 2)
        hidden_states = torch.zeros(num_Ped, 64)
        cell_states = torch.zeros(num_Ped, 64)

        value1_sum=0
        value2_sum=0
        value3_sum=0

        if self.using_cuda:
            outputs=outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            
        # For each frame in the sequence
        for framenum in range(self.args.seq_length-1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()

                nodes_abs=shift_value[framenum,node_index]+nodes_current

                nodes_abs=nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index=nodes_abs.transpose(0,1)-nodes_abs
            else:
                node_index=seq_list[framenum]>0
                nodes_current = nodes_norm[framenum,node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0,1)-corr
                nei_num_index=nei_num[framenum,node_index]

            hidden_states_current=hidden_states[node_index]
            cell_states_current=cell_states[node_index]

            input_embedded = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))

            lstm_state = self.cell.forward(input_embedded, (hidden_states_current,cell_states_current))

            for p in range(2):
                if p==0:
                    lstm_state, look = self.gcn.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn.W_nei)
                    value1, value2, value3 = look
                if p==1:
                    lstm_state, look = self.gcn1.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn1.W_nei)


            _, hidden_states_current, cell_states_current = lstm_state

            value1_sum+=value1
            value2_sum+=value2
            value3_sum+=value3

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum,node_index]=outputs_current
            hidden_states[node_index]=hidden_states_current
            cell_states[node_index] = cell_states_current

        return outputs, hidden_states, cell_states,(value1_sum/self.args.seq_length,value2_sum/self.args.seq_length,value3_sum/self.args.seq_length)


