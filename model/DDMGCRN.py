import math

import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchinfo import summary
import scipy.sparse as sp
import numpy as np
from collections import OrderedDict

class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bvc,vw->bwc', (x, A))
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self, dims, emb_dim):
        super(gcn, self).__init__()
        self.gconv = gconv_hyper()
        self.mlp = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(dims, dims)),
                         ('sigmoid1', nn.Sigmoid()),
                         ('fc2', nn.Linear(dims, emb_dim)),
                         ('sigmoid2', nn.Sigmoid()),
                         ('fc3', nn.Linear(emb_dim, emb_dim))]))

    def forward(self, x, adj):
        ho = self.gconv(x, adj)
        ho = self.mlp(ho)
        return ho

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, node_num,use_day,use_week):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.use_day=use_day
        self.use_week=use_week
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.Wdg = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.a = nn.Parameter(torch.rand(1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1, 1), requires_grad=True)
        # self.alpha1 = nn.Parameter(torch.round(torch.rand(1, 1) * 10), requires_grad=True)
        # self.alpha2 = nn.Parameter(torch.round(torch.rand(1, 1) * 10), requires_grad=True)
        # self.alpha3 = nn.Parameter(torch.round(torch.rand(1, 1) * 10), requires_grad=True)
        self.alpha1 = 3
        self.alpha2 = 3
        self.alpha3 = 3

        self.sgcn = gcn(dim_in, embed_dim)

        nn.init.xavier_normal_(self.weights_pool)
        nn.init.xavier_normal_(self.bias_pool)
        nn.init.xavier_normal_(self.Wdg)

    def forward(self, x, E_id_emb, T_D_emb, T_W_emb):

        batch_size, num_nodes, _ = T_D_emb.shape
        I = torch.eye(num_nodes).to(x.device)

        #自适应图
        AS = F.relu(torch.tanh(self.alpha1 * torch.matmul(E_id_emb, E_id_emb.transpose(0, 1))))
        AG = AGCN.get_laplacian(AS, I)
        A_hat = F.relu(torch.tanh(self.alpha2 * torch.matmul(E_id_emb, E_id_emb.transpose(0, 1))))

        # 动态图
        E = E_id_emb.unsqueeze(0).expand(batch_size, -1, -1)#(B,N,d)

        if self.use_day and self.use_week:
            Et = torch.mul(torch.mul(E, T_D_emb), T_W_emb)
        elif self.use_day and not self.use_week:
            Et = torch.mul(E, T_D_emb)
        elif not self.use_day and self.use_week:
            Et = torch.mul(E, T_W_emb)
        else:
            Et = E

        DF = self.sgcn(x, A_hat)#(B,N,d)
        DE = torch.tanh(self.alpha3 * torch.mul(Et, DF))#(B,N,d)
        DA = F.relu(torch.tanh(self.alpha3 * torch.matmul(DE, DE.transpose(1, 2))))#(B,N,N)
        #DA = AGCN.get_laplacian(DA,I)

        #自适应图卷积
        x_ag1 = torch.einsum("nm,bmi->bni", I, x)
        x_ag2 = torch.einsum("nm,bmi->bni", AG, x)
        x_ag = torch.stack([x_ag1, x_ag2], dim=1)
        weights = torch.einsum('nd,dkio->nkio', E_id_emb, self.weights_pool)
        bias = torch.matmul(E_id_emb, self.bias_pool)
        x_ag = x_ag.permute(0, 2, 1, 3)
        x_agconv = torch.einsum('bnki,nkio->bno', x_ag, weights) + bias

        # 动态图卷积
        #x_dg1 = torch.einsum("nm,bmi->bni", I, x)
        x_dg2 = torch.einsum("bnm,bmi->bni", DA, x)
        x_dgconv = torch.einsum('bni,io->bno', x_dg2, self.Wdg)
        #x_dg = torch.stack([x_dg1, x_dg2], dim=1).permute(0, 2, 1, 3)
        #x_dgconv = torch.einsum('bnki,io->bno', x_dg, self.Wdg)

        #融合
        out = self.a * x_agconv + self.b * x_dgconv

        return out

    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim,use_day,use_week):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, node_num,use_day,use_week)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, node_num,use_day,use_week)

    def forward(self, x, state, E_id_emb, T_D_emb, T_W_emb):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, E_id_emb, T_D_emb, T_W_emb))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, E_id_emb, T_D_emb, T_W_emb))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers, embed_dim,use_day,use_week):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num  # N
        self.input_dim = dim_in  # 1
        self.num_layers = num_layers  # 1
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim,use_day,use_week))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim,use_day,use_week))

    def forward(self, x, init_state, E_id_emb, T_D_emb, T_W_emb):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, E_id_emb, T_D_emb[:, t, :, :],
                                            T_W_emb[:, t, :, :])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.WQ = nn.Linear(model_dim, model_dim)
        self.WK = nn.Linear(model_dim, model_dim)
        self.WV = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)
        #self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        B, T, N, H = x.shape

        query = self.WQ(x)
        key = self.WK(x)
        value = self.WV(x)

        query = query.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        key = key.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        value = value.reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim ** 0.5
        attn_score = torch.softmax(attn_score, dim=-1)

        attention_output = attn_score @ value
        attention_output = attention_output.transpose(2, 3).reshape(B, N, T, self.model_dim)
        attention_output = self.out_proj(attention_output).permute(0, 2, 1, 3)

        attention_output = attention_output + residual

        #out = self.layer_norm(attention_output)

        return attention_output

class DMGFCRN(nn.Module):
    def __init__(
        self,
        G,
        num_nodes,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        output_dim=1,
        embed_dim=10,
        hidden_dim=64,
        cheb_k=2,
        num_heads=8,
        num_layers=1,
        dropout=0.1,
        use_day=True,
        use_week=True,
    ):
        super().__init__()
        self.G = G
        self.num_nodes = num_nodes#N
        self.in_steps = in_steps#12
        self.out_steps = out_steps#12
        self.input_dim = input_dim#1
        self.output_dim = output_dim#1
        self.embed_dim = embed_dim#
        self.hidden_dim = hidden_dim
        self.cheb_k = cheb_k
        self.num_heads = num_heads#4
        self.num_layers = num_layers#1
        self.dropout = dropout
        self.use_day = use_day
        self.use_week = use_week
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.param_dict = self.construct_param()

        self.encoder1 = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.hidden_dim, self.cheb_k, self.num_layers,
                                       self.embed_dim, self.use_day, self.use_week)

        self.encoder2 = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.hidden_dim, self.cheb_k, self.num_layers,
                                      self.embed_dim, self.use_day, self.use_week)

        self.att1 = AttentionLayer(self.hidden_dim, self.num_heads)
        self.att2 = AttentionLayer(self.hidden_dim, self.num_heads)

        self.end_conv1 = nn.Conv2d(1, self.out_steps, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, self.out_steps, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, self.out_steps, kernel_size=(1, self.hidden_dim), bias=True)

    def construct_param(self):
        param_dict = nn.ParameterDict()
        param_dict['Ed'] = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        param_dict['Day'] = nn.Parameter(torch.empty(288, self.embed_dim), requires_grad=True)
        param_dict['Week'] = nn.Parameter(torch.empty(7, self.embed_dim), requires_grad=True)
        for param in param_dict.values():
            nn.init.xavier_normal_(param)
        return param_dict

    def forward(self, x):
        # x: (B, 12, N, 3)
        x1 = x[:, :, :, 0:1]#(B, 12, N, 1)

        t_i_d_data   = x[..., 1]#(B, 12, N)
        T_D_emb = self.param_dict['Day'][(t_i_d_data * 288).type(torch.LongTensor)]#(B,12,N,d)

        d_i_w_data   = x[..., 2] #(B, 12, N)
        T_W_emb = self.param_dict['Week'][(d_i_w_data).type(torch.LongTensor)]#(B,12,N,d)

        E_id_emb = self.param_dict['Ed']  # (N,d)

        #第一层
        init_state1 = self.encoder1.init_hidden(x.shape[0])#(num_layers=1,B,N,hidden=64)
        h_en1, _ = self.encoder1(x1, init_state1, E_id_emb, T_D_emb, T_W_emb)#输出(B,12,N,64)
        h_en1 = self.att1(h_en1)
        output1 = self.dropout1(h_en1[:, -1:, :, :])
        #output1 = self.dropout1(h_en1)
        x1_out = self.end_conv1(output1)
        x1_back = self.end_conv2(output1)
        x2 = x1 - x1_back

        #第二层
        init_state2 = self.encoder2.init_hidden(x.shape[0])#(num_layers=1,B,N,hidden=64)
        h_en2, _ = self.encoder2(x2, init_state2, E_id_emb, T_D_emb, T_W_emb)#输出(B,12,N,64)
        h_en2 = self.att2(h_en2)
        output2 = self.dropout2(h_en2[:, -1:, :, :])
        #output2 = self.dropout1(h_en2)
        x2_out = self.end_conv3(output2)
        #x2_back = self.end_conv4(output2)
        #x3 = x2 - x2_back

        #第三层
        # init_state3 = self.encoder3.init_hidden(x.shape[0])#(num_layers=1,B,N,hidden=64)
        # h_en3, _ = self.encoder3(x3, init_state, E_id_emb, T_D_emb, T_W_emb)#输出(B,12,N,64)
        # output3 = self.dropout3(h_en3[:, -1:, :, :])
        # x3_out = self.end_conv5(output3)  # B, T*C, N, 1
        # x3_back = self.end_conv6(output3)
        # x4 = x3 - x3_back


        return x1_out + x2_out

