import pickle
import re
import codecs
import numpy as np
import random


'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    content = pickle.load(pkl_file)
    pkl_file.close()
    return content

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()

def get_target(target_l, clause, cause):
    """
    获取数据的BIO
    """
    cause_span = re.search(re.escape(cause), clause).span()

    start = cause_span[0]
    end = cause_span[1]
    target_l[start] = 'B'

    if start != end -1:
       for  i in range(start + 1, end):
           target_l[i] = 'I'
    return target_l


# def get_input_data(data, save_txt_path, save_data_path):
#     """
#     将数据读取和写入
#     """
#     # data = loadList(data_path)
#     outputFile1 = codecs.open(save_txt_path, 'w','utf-8') #将文本写入到csv文件
#     out_data= [] #是一个list列表，每一个list都存入一个字典
#     for index, item in enumerate(data): #对于每一个文本

#         para_data = dict() #将每一个文本写成一个字典的形式
#         write_str = '' #每一个文本吸入的内容暂时存在这个字符串里
#         content_Data = ''#每一个文本的内容
#         emotion_Data = ''#情感内容
#         target_Data = ''#标签内容
#         data_len = []#每一个子句的长度
#         docID = item[0]['docID'] #文本的ID号

#         target_d_l = [] #每一个文本的标签list
#         content_d_l = [] #每一个文本的content list

#         clause_info = item[4] #所有的子句信息
#         key_loc= int(item[2][0]['keyloc']) #情感子句所在的索引
#         emotion_word = item[2][0]['keyword'] #情感词语的内容
    
#         ec_index = [0] * len(clause_info)
#         ec_index[key_loc] = 1 #情感子句处为1，其余位置为0


#         emotion_Data = clause_info[key_loc]['content'].strip()#情感子句的内容

#         ew_index_span = re.search(re.escape(emotion_word), emotion_Data).span()
#         ew_index_l = []
#         # ec_index_l = [] #长度仍然为文本中词语的总的个数，比如相同句子中的位置
#         for indexc, itemc in enumerate(clause_info):
#             clause = itemc['content']
#             cause = itemc['cause_content']

#             clause_l = list(clause)
#             target_l = ['O'] * len(clause_l)
#             data_len.append(len(clause_l))

#             ew_index = [0] * len(clause_l)
#             if key_loc == indexc:
#                 start, end = ew_index_span[0], ew_index_span[1]
#                 for i in range(start, end):
#                     ew_index[i] = 1
#             ew_index_l.extend(ew_index) #将每个子句进行拼接，目的是在情感词语的位置赋值为1 其余的位置赋值为0

#             if cause != '':
#                 target_l = ['O'] * len(clause_l)
#                 target_l = get_target(target_l, clause, cause)
            
#             content_Data += clause
#             target_d_l.extend(target_l)
#             content_d_l.extend(clause_l)
        
#         para_data['docID'] = docID
#         para_data['emotion_Data'] =' '.joint(list(emotion_Data))
#         para_data['content_Data'] = ' '.join(content_d_l)
#         para_data['target_Data'] = ' '.join(target_d_l)
#         para_data['emotion_index'] = ew_index_l
#         para_data['data_len'] = data_len
#         para_data['ec_index'] = ec_index
#         out_data.append(para_data)

#         #将每一个文本内容写入csv文件
#         ss = ''
#         for x, y in zip(content_d_l, target_d_l):
#             ss += str(x) + '#' + str(y) + ' '

#         write_str += str(docID) + '\n' + ss.strip() + '\n' + emotion_Data + '\n\n'
#         outputFile1.write(write_str)
#     outputFile1.close()
#     saveList(out_data, save_data_path)
#     return out_data

#随机数据集的划分
def split_data(data_path, train_ratio = 0.8):
    """
    """
    data = loadList(data_path)
    data_num = len(data)
    train_num = int(data_num * train_ratio)
    index_list = list(range(data_num))
    random.shuffle(index_list)

    tr_induice = index_list[0: train_num]
    te_num = (data_num - train_num)//2
    te_induice = index_list[train_num: train_num + te_num]
    dev_induice = index_list[train_num + te_num : ]

    data_tr, data_te, data_dev = [],[],[]
    for i in range(data_num):
        if i in tr_induice:
            data_tr.append(data[i])
        elif i in te_induice:
            data_te.append(data[i])
        else:
            data_dev.append(data[i])
    return data_tr, data_te, data_dev


# data_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/data_process_eca/ECAEmnlp_new.pkl'
# data_tr_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_train.pkl'
# data_dev_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_dev.pkl'
# data_te_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_test.pkl'

# a = loadList(data_path)
# print(a[0])
# save_txt_path_tr = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_text_data_tr.csv'
# save_data_path_tr = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_seq_data_tr.pkl'

# save_txt_path_te = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_text_data_te.csv'
# save_data_path_te = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_seq_data_te.pkl'

# save_txt_path_dev = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_text_data_dev.csv'
# save_data_path_dev = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_seq_data_dev.pkl'

# data_tr, data_te, data_dev = split_data(data_path, train_ratio = 0.8)
# saveList(data_tr, data_tr_path)
# saveList(data_te, data_te_path)
# saveList(data_dev, data_dev_path)

# out_data = get_input_data(data_path, save_txt_path, save_data_path)
# print(out_data[0])
# print(data_tr[0])
# print(out_tr_data[0])