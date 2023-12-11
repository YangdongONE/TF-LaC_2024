#encoding=utf-8
import argparse
import torch
import time
import json
import numpy as np
import math
import random
from torch.autograd import Variable
import torch.nn.functional as F


class CoAtt(torch.nn.Module):
    """
    """
    def __init__(self, encoder_hidden_size):
        super(CoAtt, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.out_dense = torch.nn.Linear(self.encoder_hidden_size*6, self.encoder_hidden_size*2)
        # self.hidden_dense = torch.nn.Linear(3 * self.encoder_hidden_size, self.encoder_hidden_size)
        

    def forward(self, mask_x, mask_e, encode, encoder_e):
        # print('encoder_e = ', encoder_e.size())
        """
        mask_x: [batch, max_len][8,83] source_mask:tru false的数组 组成的【true， false】
        mask_e: [batch_max_len_e]
        encode:[max_len, batch, 2*hidden]
        encode_e:[max_len_e, batch, 2*hidden]

        hidden: torch.Size([2*layer, batch, hiddendim]) 
        hidden_e: torch.Size([2*layer, batch, hiddendim]) 

        """
        # encode_x = encode.transpose(0, 1) #[batch, max_len, 2*dim]
        # encoder_e = encoder_e.transpose(0, 1) #[batch, max_len_e, 2*dim]


        trans_e = encoder_e.transpose(1, 2) #[batch,  dim, max_len_e]
        LL= torch.bmm(encode, trans_e) # compute affinity matrix, (batch_size, context+2, question+2）
        mask_xx = torch.unsqueeze(mask_x,2) #【batch,  context+2, 1】
        mask_xxx = mask_xx.repeat(1,1,encode.size(-1)) #[batch, context+2, dim]
        mask_ee = torch.unsqueeze(mask_e,1).repeat(1, encode.size(-1), 1) #[batch, dim, question+2]
        # mask_ee = mask_ee>0
        # aa = (1-mask_ee) * torch.tensor(-1e10)
        # L = LL + aa
        maskk = torch.bmm(mask_xxx.float(), mask_ee.float()) #[batch, context+2, question+2]
        aa = maskk > 0
        cc = aa.float() 
        dd = 1- cc
        ff = dd * torch.tensor(-1e10)
        # print('LL.shape = ', LL.size())
        # print('aa.shape = ', aa.size())
        L = LL.mul(aa.float())
        L= L+ff
        L_t = torch.transpose(L, 2,1) #[batch, question+2, context+2]
        # normalize with respect to question
        a_q = F.softmax(L_t, -1)  #[batch, question+2, context+2]
        # normalize with respect to context
        a_c = F.softmax(L, -1)  #[batch, context+2, question+2]
        # summaries with respect to question, (batch_size, question+2, hidden_size)
        c_q = torch.bmm(a_q, encode) #[batch, question+2, dim]
        c_q_emb = torch.cat((encoder_e, c_q), -1) #[batch, question+2, 2*dim]
        # summaries of previous attention with respect to context
        c_d = torch.bmm(a_c, c_q_emb) #[batch, context+2, 2* dim]
        # final coattention context, (batch_size, context+1, 3*hidden_size)
        co_att = torch.cat((encode, c_d), -1)  #[batch, context+2, 3* dim]
        mask_out = mask_xx.repeat(1,1,co_att.size(-1))#将实验结构进一步mask
        # mask_out = mask_xx.repeat(1,1,co_att.size(-1))
        co_att.mul(mask_out.float())
        # co_att = self.

        #获取对应的隐藏层，在这里隐藏层加入了情感的拼接
        # hidden_co = torch.cat((hidden, hidden_e), -1)  #[2* layers, batch, 2* dim]
        # hidden_co = torch.cat((hidden, hidden_co), -1)  #[2* layers, batch, 3* dim]
        # hidden_co = self.hidden_dense(hidden_co)#[2* layers, batch, dim]

        # co_att = co_att.transpose(0,1) #[context+2, batch,  3* dim]
        co_att = self.out_dense(co_att)
        # print('co_att.shape = ', co_att.size())

        return co_att