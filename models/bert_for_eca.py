import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers.modeling_bert import BertPreTrainedModel
from .transformers.modeling_bert import BertModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from .layers.GRU_decoder import Decoder
from .layers.crf import CRF
from .layers.AttLayer import CoAtt
from .layers.AttLayer_sing import Att
# from losses.loss_func import CELoss, SupConLoss,DualLoss


from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from tools.finetuning_argparse_eca import get_argparse #需要修改一个参数，或者重新写一个类
from torch.autograd import Variable
from einops import rearrange

args = get_argparse().parse_args()

class Bert2Crf(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert2Crf, self).__init__(config)
        self.bert = BertModel(config)
        if (args.model_encdec == 'bert2gruAtt' or args.model_encdec ==  'bert2crfAtt') and  args.att_type == 'CoAtt':
            self.Attlayer = CoAtt(encoder_hidden_size = args.encoder_hidden_size)
        if (args.model_encdec == 'bert2gruAtt' or args.model_encdec ==  'bert2crfAtt')  and args.att_type == 'Att':
            self.Attlayer = Att(encoder_hidden_size = args.encoder_hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, args.label_size)      # 因为单词向量两两拼接，所以要乘二 ，如果不采用拼接的方法而使用相加的方法，则不用乘二
        self.crf = CRF(num_tags=args.label_size, batch_first=True)

        # loss = nn.CrossEntropyLoss()
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, crf_attention_mask=None,labels=None, testing = False):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        cls_out_x = sequence_output[:,0,:] #[batch, dim],第0个向量CLS

        # concatenate vectors of w_i and w_j
        # tensor_empty = torch.empty(1,sequence_output.shape[1]*sequence_output.shape[1],sequence_output.shape[2]*2)
        # for i in range(sequence_output.shape[1]):
        #     for j in range(sequence_output.shape[1]):
        #         # 通过运行一个简单的例子可以证明，按照如下方式生成的张量内部各“子张量”的顺序，与直接应用reshape或view生成的张量顺序一致
        #         tensor_empty[:,i*sequence_output.shape[1]+j,:] = torch.cat((sequence_output[:,i,:], sequence_output[:,j,:]), 1)

        # 向量相加
        # expand_sequence_output = sequence_output.expand(sequence_output.shape[0],sequence_output.shape[1],sequence_output.shape[1],sequence_output.shape[2])
        # expand_sequence_output = torch.add(torch.transpose(expand_sequence_output,1,2),expand_sequence_output)
        # add_sequence_output = expand_sequence_output.reshape(sequence_output.shape[0],sequence_output.shape[1]**2,sequence_output.shape[2]).cuda()

        # start_time = time.time()
        # 向量拼接
        expand_output_1 = sequence_output.unsqueeze(1).expand(sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[1], sequence_output.shape[2]).transpose(1, 2)
        expand_output_2 = sequence_output.unsqueeze(2).expand(sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[1], sequence_output.shape[2])
        cat_sequence_output = torch.cat((expand_output_1, expand_output_2), dim=3)
        cat_sequence_output = cat_sequence_output.reshape(sequence_output.shape[0], sequence_output.shape[1]**2, sequence_output.shape[2]*2)
        # end_time = time.time()
        # print("concatenating tensor:", end_time-start_time)

        # logits = self.classifier(sequence_output)

        # logits = self.classifier(add_sequence_output)

        # predict_start = time.time()
        logits = self.classifier(cat_sequence_output)
        outputs = (logits,)
        if labels is not None:
            labels = labels.reshape(sequence_output.shape[0], -1)
            loss = self.crf(emissions=logits, tags=labels, mask=crf_attention_mask)
            outputs = (-1*loss,)+outputs

        return outputs # (loss), scores

class Bert2Gru(BertPreTrainedModel):

    def __init__(self, config):
        super(Bert2Gru, self).__init__(config)
        self.bert = BertModel(config)
        if (args.model_encdec == 'bert2gruAtt' or args.model_encdec ==  'bert2crfAtt') and  args.att_type == 'CoAtt':
            self.Attlayer = CoAtt(encoder_hidden_size = args.encoder_hidden_size)
        if (args.model_encdec == 'bert2gruAtt' or args.model_encdec ==  'bert2crfAtt')  and args.att_type == 'Att':
            self.Attlayer = Att(encoder_hidden_size = args.encoder_hidden_size)
        # self.bert_e = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.decoder = Decoder(args, num_classes=args.label_size, dropout=0.2)
        self.clsdense = nn.Linear(config.hidden_size, args.decoder_hidden_size)
        # self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None,input_e_ids = None, token_type_e_ids=None, attention_e_mask=None, labels=None, testing = False):
        #注意这里的target是根据长度排序过的数据
        x_mask = attention_mask > 0 #score: 原始数据，是词语的index [8,83] source_mask:tru false的数组 组成的【true， false】
        x_len = torch.sum(x_mask.int(), -1)
        # x = x.transpose(0, 1) #原始数据变成了【83， 8】
        target_= labels # 是一个PackedSequence的数据
        max_len = input_ids.size(1) #in other sq2seq, max_len should be target.size() batch——size
        batch_size = input_ids.size(0) 

        if labels != None:
            target, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, total_length=max_len) #target [83, 8]

        label_size = args.label_size
        outputs =  Variable(torch.zeros(max_len, batch_size, label_size)).cuda()
        attention = Variable(torch.zeros(max_len, batch_size, max_len)).cuda()

        #对x进行编码
        outputs_x =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        encoder_outputs = outputs_x[0]
        cls_out = encoder_outputs[:,0,:] #[batch, dim],第0个向量CLS
        encoder_outputs = self.dropout(encoder_outputs) #[batch, max_len, dim]
        x_mask = x_mask.transpose(0,1)
        hidden =  cls_out.unsqueeze(0).repeat(args.decoder_num_layers, 1, 1)# [args.decoder_num_layers, batch, hiddendim]
        hidden = self.clsdense(hidden)

        #对xe进行编码
        if input_e_ids is not None:
            xe_mask = attention_e_mask > 0 #score: 原始数据，是词语的index [8,83] source_mask:tru false的数组 组成的【true， false】
            # xe_mask = xe_mask.transpose(0,1)
            xe_len = torch.sum(xe_mask.int(), -1)
            # x_e = attention_e_mask.transpose(0, 1) #原始数据变成了【83， 8】
            outputs_e =self.bert(input_ids = input_e_ids,attention_mask=attention_e_mask,token_type_ids=token_type_e_ids)
            encoder_outputs_e = outputs_e[0]
            cls_out_e = encoder_outputs_e[:,0,:] #[batch, dim],第0个向量CLS
            encoder_outputs_e = self.dropout(encoder_outputs_e) #[batch, max_len, dim]

            hidden_e =  cls_out_e.unsqueeze(0).repeat(args.decoder_num_layers, 1, 1) # [args.decoder_num_layers, batch, hiddendim]
            hidden_e = self.clsdense(hidden_e)
            # hidden = hidden[:self.args.decoder_num_layers] # [args.decoder_num_layers, batch, hiddendim]
            # hidden_e = hidden_e[:self.args.decoder_num_layers] # [args.decoder_num_layers, batch, hiddendim]
            # print('args.model_encdec ==', args.model_encdec)
            encoder_outputs = self.Attlayer(attention_mask, attention_e_mask, encoder_outputs, encoder_outputs_e) #[max_len, batch, dim]
        

        output = Variable(torch.zeros((batch_size))).long().cuda()
        encoder_outputs = encoder_outputs.transpose(0,1)
        # hidden = hidden.unsqueeze(0).repeat(args.decoder_num_layers,  1, 1)

        for t in range(max_len):
            current_encoder_outputs = encoder_outputs[t,:,:].unsqueeze(0) #[1, batch, 2*encoder_hidden_size] 第t个词语的表示，去掉第一个维度，【batch, 2*dim】
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_outputs, current_encoder_outputs, t, max_len, x_mask)
            outputs[t] = output
            attention[t] = attn_weights.squeeze()
            #is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if testing:
                output = Variable(top1).cuda()
            else:
                output = Variable(target[t]).cuda()

        if testing:
            outputs = outputs.transpose(0,1)
            return outputs, attention
        else:
            packed_y = torch.nn.utils.rnn.pack_padded_sequence(outputs, x_len.to('cpu'))
            score  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y.data), target_.data)
            return score, outputs

class Bert2Softmax(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert2Softmax, self).__init__(config)
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, args.label_size)
        # self.classifier = nn.Linear(config.hidden_size,args.label_size)
        self.pre_concat_fc1 = nn.Linear(config.hidden_size,config.hidden_size)
        self.pre_concat_fc2 = nn.Linear(config.hidden_size,config.hidden_size)
        self.classifier_emo = nn.Linear(config.hidden_size, args.emotion_label_size)
        self.classifier_cause = nn.Linear(config.hidden_size, args.cause_label_size)
        # self.linear_main_sub = nn.Linear(args.label_size,2)
        self.conv1 = nn.Conv2d(
            in_channels=config.hidden_size*2,
            out_channels=config.hidden_size*2,
            kernel_size=(1,1),
            padding=0
        )
        self.norm1 = nn.LayerNorm(config.hidden_size*2, 1e-12)
        self.conv2 = nn.Conv2d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size * 2,
            kernel_size=(3,3),
            padding=1
        )
        self.norm2 = nn.LayerNorm(config.hidden_size * 2, 1e-12)
        self.conv3 = nn.Conv2d(
            in_channels=config.hidden_size*2,
            out_channels=config.hidden_size*2,
            kernel_size=(1,1),
            padding=0
        )
        self.norm3 = nn.LayerNorm(config.hidden_size*2, 1e-12)
        self.conv4 = nn.Conv2d(
            in_channels=config.hidden_size*2,
            out_channels=config.hidden_size*2,
            kernel_size=(1,1),
            padding=0
        )
        self.norm4 = nn.LayerNorm(config.hidden_size*2, 1e-12)
        self.conv5 = nn.Conv2d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size * 2,
            kernel_size=(3,3),
            padding=1
        )
        self.norm5 = nn.LayerNorm(config.hidden_size * 2, 1e-12)
        self.conv6 = nn.Conv2d(
            in_channels=config.hidden_size*2,
            out_channels=config.hidden_size*2,
            kernel_size=(1,1),
            padding=0
        )
        self.norm6 = nn.LayerNorm(config.hidden_size*2, 1e-12)

        self.fuse_emo_cau = nn.Linear(config.hidden_size*4, config.hidden_size*2)
        self.diag_linear = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, emotion_labels=None, cause_labels=None,sub_emotion_labels=None,sub_cause_labels=None,
                testing=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:,1:,:]
        sub_sequence_output = outputs[0][:,10:,:]
        # polarity_output = outputs[0][:,1:10,:]
        cls_output = outputs[0][:,0,:]
        sequence_output = self.dropout(sequence_output)

        sequence_output1 = self.pre_concat_fc1(sequence_output)
        sequence_output1 = F.relu(sequence_output1)
        sequence_output2 = self.pre_concat_fc2(sequence_output)
        sequence_output2 = F.relu(sequence_output2)

        expand_output_1 = sequence_output1.unsqueeze(1).expand(sequence_output1.shape[0], sequence_output1.shape[1], sequence_output1.shape[1], sequence_output1.shape[2])
        expand_output_2 = sequence_output2.unsqueeze(1).expand(sequence_output2.shape[0], sequence_output2.shape[1], sequence_output2.shape[1], sequence_output2.shape[2])
        # 向量拼接
        cat_sequence_output = torch.cat((expand_output_1.permute(0,2,1,3), expand_output_2), dim=3)

        x = rearrange(cat_sequence_output, 'b m n d -> b d m n')
        x = self.conv_forward(x, self.conv1, self.norm1)
        x = self.conv_forward(x, self.conv2, self.norm2)
        x = self.conv_forward(x, self.conv3, self.norm3)
        x = rearrange(x, 'b d m n -> b m n d')
        cat_sequence_output = x + cat_sequence_output
        x = rearrange(cat_sequence_output, 'b m n d -> b d m n')
        x = self.conv_forward(x, self.conv4, self.norm4)
        x = self.conv_forward(x, self.conv5, self.norm5)
        x = self.conv_forward(x, self.conv6, self.norm6)
        x = rearrange(x, 'b d m n -> b m n d')
        cat_sequence_output = x + cat_sequence_output

        logits = self.classifier(cat_sequence_output)
        logits_emotion = self.classifier_emo(sub_sequence_output)
        logits_cause = self.classifier_cause(sub_sequence_output)
        logits_diag = torch.diagonal(cat_sequence_output[:,9:,9:,:],dim1=1,dim2=2).transpose(1,2)
        # logits_diag = self.diag_linear(logits_diag)
        # logits_diag = F.relu(logits_diag)
        logits_diag = F.normalize(logits_diag,dim=-1)

        logits_emotion_main = cat_sequence_output[:, 2, 9:, :]
        logits_cause_main = cat_sequence_output[:, 1, 9:, :]

        logits_emotion_main_rescale = torch.cat((F.softmax(logits_emotion_main,dim=-1)[:,:,0:1]*0.5,
                                                 F.softmax(logits_emotion_main,dim=-1)[:,:,1:]),dim=-1)
        logits_cause_main_rescale = torch.cat((F.softmax(logits_cause_main,dim=-1)[:,:,0:1]*0.5,
                                                 F.softmax(logits_cause_main,dim=-1)[:,:,1:]),dim=-1)
        logits_emotion_cause_boundary = (logits_emotion_main_rescale + logits_cause_main_rescale)

        logits_emotion_cause_boundary = F.normalize(logits_emotion_cause_boundary,dim=-1)

        outputs = (logits,logits_emotion,logits_cause)
        if labels is not None and emotion_labels is not None and cause_labels is not None:

            loss_pair = F.cross_entropy(logits.reshape(logits.shape[0]*logits.shape[1]*logits.shape[1], args.label_size), labels.reshape(labels.shape[0]*labels.shape[1]*labels.shape[1]),ignore_index=-1)
            loss_emo = F.cross_entropy(logits_emotion.reshape(logits_emotion.shape[0]*logits_emotion.shape[1],args.emotion_label_size),
                                       sub_emotion_labels.reshape(sub_emotion_labels.shape[0]*sub_emotion_labels.shape[1]),ignore_index=-1)
            loss_cau = F.cross_entropy(logits_cause.reshape(logits_cause.shape[0]*logits_cause.shape[1],args.cause_label_size),
                                       sub_cause_labels.reshape(sub_cause_labels.shape[0]*sub_cause_labels.shape[1]),ignore_index=-1)

            boundary_diag_sub_kl = F.kl_div(F.log_softmax(logits_diag.reshape(logits_diag.shape[0]*logits_diag.shape[1],-1),dim=-1),
                                         F.softmax(logits_emotion_cause_boundary.reshape(logits_emotion_cause_boundary.shape[0]*logits_emotion_cause_boundary.shape[1],-1),dim=-1))
            boundary_sub_diag_kl = F.kl_div(F.log_softmax(logits_emotion_cause_boundary.reshape(logits_emotion_cause_boundary.shape[0]*logits_emotion_cause_boundary.shape[1],-1),dim=-1),
                                         F.softmax(logits_diag.reshape(logits_diag.shape[0]*logits_diag.shape[1],-1),dim=-1))
            loss = loss_pair + loss_emo + loss_cau + 0.4*boundary_sub_diag_kl + 0.4*boundary_diag_sub_kl
            outputs = (loss,) + outputs
        return outputs  # (loss), scores

    def conv_forward(self, x, conv, norm):
        x = conv(x)
        n = x.size(-1)
        x = rearrange(x, 'b d m n -> b (m n) d')
        x = norm(x)
        x = F.relu(x)
        x = rearrange(x, 'b (m n) d -> b d m n', n=n)
        return x
