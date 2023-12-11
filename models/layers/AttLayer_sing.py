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



class Att(torch.nn.Module):
    """
    """
    def __init__(self, encoder_hidden_size):
        super(Att, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        # self.out_dense = torch.nn.Linear(self.encoder_hidden_size*6, self.encoder_hidden_size*2)
        # self.hidden_dense = torch.nn.Linear(2 * self.encoder_hidden_size, self.encoder_hidden_size)
        
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
        #注意力矩阵
        LL= torch.bmm(encode, trans_e) # compute affinity matrix, (batch_size, context+2, question+2）
        # print('LL = ', LL.size())
        #mask 矩阵
        # mask_xx = torch.unsqueeze(mask_x,2) #【batch,  context+2, 1】
        # mask_xxx = mask_xx.repeat(1,1,encode_x.size(-1)) #[batch, context+2, dim]
        # mask_ee = torch.unsqueeze(mask_e,1).repeat(1, encode_x.size(-1), 1) #[batch, dim, question+2]
        # mask_ee = mask_ee>0
        # aa = (1-mask_ee) * torch.tensor(-1e10)
        # L = LL + aa
        # maskk = torch.bmm(mask_xxx.float(), mask_ee.float()) #[batch, context+2, question+2]
        # aa = maskk > 0 #获取bool型，在非填充的位置上是true # [batch, context+2, question+2]
        # cc = aa.float()  #转换为float类型，将true类型转换为1， false类型转换为0  #[batch, context+2, question+2]
        # dd = 1- cc #将填充的位置转换为1，非填充位置转换为0   #[batch, context+2, question+2]
        # ff = dd * torch.tensor(-1e10) #将填充位置转换为 -1000000000000.0    #[batch, context+2, question+2]

        # LL = LL + ff #将填充位置转换为很小的数字
        sumL = torch.sum(LL, -1)#[batch, contex_len]
        #先进行mask，然后进行归一化,将为0的位置变为-1e12
        maskx = mask_x.float() #[batch, content_len],填充位置为0，非填充位置为1
        aa = (1-maskx) * torch.tensor(1e-10) #填充位置为1e10，非填充位置为0
        # print('sumL = ', sumL)
        sumL = sumL + aa
        # print('sumLL = ', sumL)
        # LL = F.softmax(sumL, 1) #【batch, context】情感词语对每个词语的权重，做了归一化，每一个词语对context中的每个词语的权重和为1
        sumLmax = torch.max(sumL, -1)[0].unsqueeze(1)
        # print('sumLmax = ', sumLmax)
        norma_att = torch.div(sumL, sumLmax)

        norma_att = norma_att.unsqueeze(2).repeat(1,1,encode.size(-1))
        outputs = encode.mul(norma_att) + encode
        # outputs = encode_x.mul(LL)

        # outputs = outputs.transpose(0,1) #[context+2, batch,  dim]
        # co_att = self.out_dense(co_att)

        return outputs