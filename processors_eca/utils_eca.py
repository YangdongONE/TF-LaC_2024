import csv
import json
import torch
from models.transformers import BertTokenizer
import pickle
import codecs
import re
import string
import copy
from processors_eca.func import loadList, saveList, get_clean_data_ch, get_clean_data_en

class EcaTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_en_pkl(self, data_path, save_csv_path=None):
        # 获取每个数据的属性
        """
        要获取文本数据， 情感数据， 原因的位置，以及要进行核对原因位置是否正确
        获取example的list列表
        """
        """
        s = "string. With. Punctuation?"
        table = str.maketrans({key: ' ' + key + ' ' for key in string.punctuation})
        print('table = ', table)
        new_s = s.translate(table)
        [
        {'docId': 0}, 
        {'name': 'surprise', 'value': '5'}, 
        [{'keyword': 'was startled by', 'keyloc': 1, 'clauseID': 2}], 
        [{'index': 1, 'cause_content': 'his unkempt hair and attire', 'clauseID': 2}], 
        [{'cause': 'N', 'id': '1', 'keywords': 'N', 'clauseID': 1, 'content': 'That day Jobs walked into the lobby of the video game manufacturer Atari and told the personnel director'}, 
        {'cause': 'Y', 'id': '2', 'keywords': 'Y', 'clauseID': 2, 'content': 'who was startled by his unkempt hair and attire', 'cause_content': 'his unkempt hair and attire', 'key_content': 'was startled by'}, {'cause': 'N', 'id': '3', 'keywords': 'N', 'clauseID': 3, 'content': "that he wouldn't leave until they gave him a job."}]]
        """
        return_data = []
        data = loadList(data_path)

        for index, item in enumerate(data):

            example_dic = dict()
            docID = item[0]['docId']

            # 修正英文数据集之错误
            if docID == 786:
                item[4][1]['content'] = 'Possibly M de Coralth was the cause of her strange disquietude'
            if docID == 1923:
                item[4][3]['content'] = 'The young peasant himself was still more astonished'
            if docID == 1863:
                item[4][18]['content'] = "' said Alice in a tone of great surprise `Of course not"
            if docID == 553:
                item[4][9]['key_content'] = 'surprised'
                item[2][0]['keyword'] = 'surprised'

            emotion_loc = int(item[2][0]['keyloc'])
            clause_info = item[4]  # clause 信息
            category_name = item[1]['name']  # 文档情绪词所属极性

            emo_clause = clause_info[emotion_loc]['content']
            emotion_content = get_clean_data_en(emo_clause).split()

            example_dic['docID'] = docID
            example_dic['emo_data'] = emotion_content

            content_data = []
            # target_data = []
            clause_len = []

            for indexc, itemc in enumerate(clause_info):
                content_text = get_clean_data_en(itemc['content'])
                content_l = content_text.split()
                content_l.append('[SEP]')  # 添加【SEP】字符
                content_data.extend(content_l)  # 添加子句的word
                clause_len.append(len(content_l))  # 获取子句的长度

            content_len = len(content_data)  # 获取总文本之长度
            target_matrix = [['O'] * content_len for _ in
                             range(content_len)]  # 初始化标签全为O的矩阵,须以迭代之方式生成此嵌套列表，否则赋值会对一整个内层列表进行操作

            cause_range = []  # 所有的原因片段的起始位置，是考虑到一段文本中有多个原因片段的情况
            emo_range = []  # 所有的情绪片段的起始位置，尽管该数据集中每段文本均只有一个情绪片段，但多个情绪片段是更通用的情况

            for indexc, itemc in enumerate(clause_info):
                ifcause = itemc['cause']
                # content_text = get_clean_data_ch(itemc['content'])
                if ifcause == 'Y':  # 原因标签
                    cause_content = get_clean_data_en(itemc['cause_content'])
                    start, end = get_en_target(content_data, cause_content)
                    cause_range.append((start, end))

                    for i in range(start, end):
                        for j in range(start, end):
                            target_matrix[i][j] = 'C'  # 此四个赋值代码之顺序不可随意更改，虽有重复赋值之部分
                            # target_matrix[j][i] = 'C'   # 此一句实为多余
                            target_matrix[i][i] = 'I-C'
                    target_matrix[start][start] = 'B-C'

                ifemo = itemc['keywords']  # 情绪标签
                if ifemo == 'Y':

                    # emo_start = int(item[2][0]['key-words-begin']) + sum(clause_len[:indexc])
                    # emo_end = emo_start + int(item[2][0]['keywords-length'])
                    keyword_content = get_clean_data_en(item[2][0]['keyword'])
                    # print("当前文档之ID：",docID)

                    # if docID == 786:
                    #     print(keyword_content)
                    # if docID == 1923:
                    #     print(keyword_content)
                    # if docID == 1863:
                    #     print(keyword_content)
                    # if docID == 553:
                    #     print(keyword_content)

                    emo_start, emo_end = get_en_target(content_data, keyword_content)
                    emo_range.append((emo_start, emo_end))

                    for i in range(emo_start, emo_end):
                        for j in range(emo_start, emo_end):
                            target_matrix[i][j] = 'E'
                            target_matrix[i][i] = 'I-E'
                    target_matrix[emo_start][emo_start] = 'B-E'

                # 极性标签
                for single_cause in cause_range:
                    cause_start, cause_end = single_cause
                    for i in range(cause_start, cause_end):
                        for single_emo in emo_range:
                            emotion_start, emotion_end = single_emo
                            for j in range(emo_start, emo_end):
                                target_matrix[i][j] = category_name
                                target_matrix[j][i] = category_name
            example_dic['content_data'] = content_data
            example_dic['target_data'] = target_matrix
            example_dic['clause_len'] = clause_len
            example_dic['content_len'] = len(content_data)

            return_data.append(example_dic)

        # for i in range(0, 3):
        #     dd = return_data[i]
        #     print(dd['content_data'])
        #     print(dd['target_data'])
        #     print('核对数据')
        return return_data

    def _read_ch_pkl(self, data_path, save_csv_path = None):
        #获取每个数据的属性
        """
        要获取文本数据， 情感数据， 原因的位置，以及要进行核对原因位置是否正确
        获取example的list列表
        """
        #[{'docID': 0}, 
        # {'name': 'happiness', 'value': '3'}, 
        # [{'key-words-begin': '0', 'keywords-length': '2', 'keyword': '激动', 'clauseID': 3, 'keyloc': 2}], 
        # [{'id': '1', 'type': 'v', 'begin': '43', 'length': '11', 'index': 1, 'cause_content': '接受并采纳过的我的建议', 'clauseID': 5}], 
        
        # [{'id': '1', 'cause': 'N', 'keywords': 'N', 'clauseID': 1, 'content': '河北省邢台钢铁有限公司的普通工人白金跃，', 'cause_content': '', 'dis': -2}, 
        # {'id': '2', 'cause': 'N', 'keywords': 'N', 'clauseID': 2, 'content': '拿着历年来国家各部委反馈给他的感谢信，', 'cause_content': '', 'dis': -1}, 
        # {'id': '3', 'cause': 'N', 'keywords': 'Y', 'clauseID': 3, 'content': '激动地对中新网记者说。', 'cause_content': '', 'dis': 0}, 
        # {'id': '4', 'cause': 'N', 'keywords': 'N', 'clauseID': 4, 'content': '“27年来，', 'cause_content': '', 'dis': 1}, 
        # {'id': '5', 'cause': 'Y', 'keywords': 'N', 'clauseID': 5, 'content': '国家公安部、国家工商总局、国家科学技术委员会科技部、卫生部、国家发展改革委员会等部委均接受并采纳过的我的建议', 'cause_content': '接受并采纳过的我的建议', 'dis': 2}]]
        
        return_data = []
        data = loadList(data_path)

        for index, item in enumerate(data):
            example_dic =dict()
            docID = item[0]['docID']
            emotion_loc = int(item[2][0]['keyloc'])
            clause_info = item[4]       # clause 信息
            category_name = item[1]['name']         # 文档情绪词所属极性


            emo_clause = clause_info[emotion_loc]['content']
            emotion_content = list(get_clean_data_ch(emo_clause))

            example_dic['docID'] = docID
            example_dic['emo_data'] = emotion_content
            
            content_data = []
            # target_data = []
            clause_len = []

            for indexc, itemc in enumerate(clause_info):
                content_text =get_clean_data_ch(itemc['content'])
                content_l = list(content_text)
                content_l.append('[SEP]')#添加【SEP】字符
                content_data.extend(content_l) #添加子句的word
                clause_len.append(len(content_l))#获取子句的长度

            content_len = len(content_data) #获取总文本之长度
            target_matrix = [['O'] * content_len for _ in range(content_len)] # 初始化标签全为O的矩阵,须以迭代之方式生成此嵌套列表，否则赋值会对一整个内层列表进行操作
            # 生成情绪片段序列标注
            emotion_label = ['O'] * content_len
            # 生成原因片段序列标注
            cause_label = ['O'] * content_len

            cause_range = []                              # 所有的原因片段的起始位置，是考虑到一段文本中有多个原因片段的情况
            emo_range = []                                # 所有的情绪片段的起始位置，尽管该数据集中每段文本均只有一个情绪片段，但多个情绪片段是更通用的情况

            emotion_label_in_mtrx = ['O'] * content_len
            cause_label_in_mtrx = ['O'] * content_len

            for indexc, itemc in enumerate(clause_info):
                ifcause = itemc['cause']
                # content_text = get_clean_data_ch(itemc['content'])
                if ifcause == 'Y':                                          # 原因标签
                    cause_content = get_clean_data_ch(itemc['cause_content'])
                    start, end = get_ch_target(content_data, cause_content)
                    cause_range.append((start,end))

                    for i in range(start,end):
                        for j in range(start,end):
                            target_matrix[i][j] = 'C'                       # 此四个赋值代码之顺序不可随意更改，虽有重复赋值之部分
                            # target_matrix[j][i] = 'C'   # 此一句实为多余
                            target_matrix[i][i] = 'I-C'
                    target_matrix[start][start] = 'B-C'

                ifemo = itemc['keywords']                                   # 情绪标签
                if ifemo == 'Y':
                    emo_start = int(item[2][0]['key-words-begin']) + sum(clause_len[:indexc])
                    emo_end = emo_start + int(item[2][0]['keywords-length'])
                    emo_range.append((emo_start, emo_end))

                    for i in range(emo_start, emo_end):
                        for j in range(emo_start, emo_end):
                            target_matrix[i][j] = 'E'
                            target_matrix[i][i] = 'I-E'
                    target_matrix[emo_start][emo_start] = 'B-E'

                # 极性标签
                for single_cause in cause_range:
                    cause_start, cause_end = single_cause
                    for i in range(cause_start, cause_end):
                        for single_emo in emo_range:
                            emotion_start, emotion_end = single_emo
                            for j in range(emo_start, emo_end):
                                target_matrix[i][j] = category_name
                                target_matrix[j][i] = category_name

                        # 情绪片段序列
                            for k in range(emotion_start, emotion_end):
                                emotion_label[k] = 'I-ES'
                                emotion_label_in_mtrx[k] = 'I-ES'
                            emotion_label[emotion_start] = 'B-ES'
                            emotion_label_in_mtrx[emotion_start] = 'B-ES'

                        # 原因片段序列
                            for k in range(cause_start, cause_end):
                                cause_label[k] = 'I-CS'
                                cause_label_in_mtrx[k] = 'I-CS'
                            cause_label[cause_start] = 'B-CS'
                            cause_label_in_mtrx[cause_start] = 'B-CS'

            origin_len = len(target_matrix)
            new_len = origin_len + 9    #9是O，C，E和六个极性的长度
            cause_elements2insert = ['O','C','O','O','O','O','O','O','O']
            emotion_elements2insert = ['O','O','E','O','O','O','O','O','O']

            for single_line_index in range(len(target_matrix)): #每一行前面都要加9个标签
                target_matrix[single_line_index][0:0] = ['O','O','O','O','O','O']  #六个极性对应的标签
                target_matrix[single_line_index][0:0] = [emotion_label_in_mtrx[single_line_index]] # 标签E对应的标签
                target_matrix[single_line_index][0:0] = [cause_label_in_mtrx[single_line_index]] #  C对应的标签
                target_matrix[single_line_index][0:0] = ['O'] #O对应的标签
            six_label2insert = [['O'] * new_len for _ in range(6)]
            plrt_list = ['O', 'O', 'O', 'O', 'O', 'O']
            plrt_dict = {'anger':0,'disgust':1,'fear':2,'happiness':3,'sadness':4,'surprise':5}
            for pol in [category_name]:
                plrt_list[plrt_dict[pol]] = pol
            for idx in range(len(six_label2insert)):
                six_label2insert[idx][idx+3] = plrt_list[idx]
            O_label2insert = [['O'] * new_len]

            cause_label_in_mtrx[0:0] = cause_elements2insert
            emotion_label_in_mtrx[0:0] = emotion_elements2insert
            target_matrix[0:0] = six_label2insert
            target_matrix[0:0] = [emotion_label_in_mtrx]
            target_matrix[0:0] = [cause_label_in_mtrx]
            target_matrix[0:0] = O_label2insert

            example_dic['content_data'] = content_data
            example_dic['target_data'] = target_matrix
            example_dic['clause_len'] = clause_len
            example_dic['content_len'] = len(content_data)
            example_dic['emotion_labels'] = emotion_label
            example_dic['cause_labels'] = cause_label
            return_data.append(example_dic)
        
        # for i in range(0, 3):
        #     dd = return_data[i]
        #     print(dd['content_data'])
        #     print(dd['target_data'])
        #     print('核对数据')
        return return_data
    def _read_merge_ch_pkl(self,data_path,save_csv_path = None):
        """
            要获取文本数据， 情感数据， 原因的位置，以及要进行核对原因位置是否正确
            获取example的list列表
        """
        data = loadList(data_path)
        return_data = []

        for index, item in enumerate(data):
            example_dict = dict()
            content_l = [] # 保存此文档中的所有子句（包括[sep]）
            emo_cau_list = [] #用于保存label一列的标注内容,这个list里面元素的个数一定与以下cause_loc_list中的元素个数相同
            cause_loc_list = [] #用于保存当前label对应的原因子句的位置，虽然片段级别的任务在标注时用不到这个子句的位置，但是说不定模型可以用得上
            clause_len = [] #保存每一个子句的长度（[sep]也占一个长度）
            docID = item.iloc[0]['id'].split(' ')[0]
            clause_num = int(item.iloc[0]['id'].split(' ')[1])

            # emo_cause_loc = item.iloc[1]['id'] #可能会存在多组
            # emo_loc = int(emo_cause_loc.split(',')[0].replace('(',''))
            # cause_loc = int(emo_cause_loc.split(',')[1].replace(')',''))
            for i in range(2,2+clause_num):  #先把每个样例的文本和标签保存下来，文本的每个字都分开，加2是因为文档内容从下标2的地方开始
                if str(item.iloc[i]['text_a'])!='nan':
                    a_clause = item.iloc[i]['text_a']
                    a_clause = list(a_clause.replace(' ',''))
                    a_clause.append('[SEP]')
                    clause_len.append(len(a_clause))
                    content_l.extend(a_clause)
                if str(item.iloc[i]['label'])!='nan':  # 不便在这里先把用&作区隔的多标签情况处理一下
                    emo_cau_list.append(item.iloc[i]['label'])
                    cause_loc_list.append(int(item.iloc[i]['id']))
            content_len = sum(clause_len) # 保存一整个文档的长度（包含[SEP]）
            # 生成情绪原因片段对的标签矩阵
            target_matrix = [['O'] * content_len for _ in range(content_len)]  # 初始化标签全为O的矩阵,须以迭代之方式生成此嵌套列表，否则赋值会对一整个内层列表进行操作
            # 生成情绪片段序列标注
            emotion_label = ['O'] * content_len
            # 生成原因片段序列标注
            cause_label = ['O'] * content_len

            # 生产标签矩阵里面的子任务标注
            emotion_label_in_mtrx = ['O'] * content_len
            cause_label_in_mtrx = ['O'] * content_len

            #根据label一列，对文本进行标注，同时把情绪词所在的子句保存下来（可能有多个）
            # 然后把content_l中各个“分散”的子句列表，整合成一个列表中的一个完整句子
            # 初步想了想，还是应该把整个文档作为一个整体，然后对整个标签矩阵处理比较好
            polar_label = []
            for j in range(len(emo_cau_list)):
                cause_loc = cause_loc_list[j]
                emo_cause_l = emo_cau_list[j].split('&') # 这一步得到的仍是一个列表 #要考虑到存在用&作区隔的情况
                for emo_cause in emo_cause_l:
                    emo_loc = int(emo_cause.split('-')[0]) #情绪所在的行数，用来获取极性
                    emo_text = get_clean_data_ch(emo_cause.split('-')[1])
                    cause_text = get_clean_data_ch(emo_cause.split('-')[2]) # 先把所有的标点符号去除
                    if str(item.iloc[emo_loc+1]['emo_eng']) == 'nan':
                        raise ValueError('对应到了空的情绪')
                    polar_list = item.iloc[emo_loc+1]['emo_eng'].split('&')  #加一而不加二
                    emo_ch_list = item.iloc[emo_loc+1]['emo_ch'].split('&')
                    plrt = polar_list[0]
                    if emo_text in emo_ch_list:
                        plrt = polar_list[emo_ch_list.index(emo_text)]
                    else:
                        print("标注的情绪词无法和真实情绪词对应")
                        raise ValueError("标注的情绪词无法和真实情绪词对应")
                    polar_label.append(plrt)
                    cau_start, cau_end = get_ch_target(content_l, cause_text)
                    emo_start, emo_end = get_ch_target(content_l, emo_text)

                    for k in range(cau_start, cau_end):
                        for v in range(cau_start, cau_end):
                            target_matrix[k][v] = 'C'
                            target_matrix[k][k] = 'I-C'
                    target_matrix[cau_start][cau_start] = 'B-C'

                    for k in range(emo_start, emo_end):
                        for v in range(emo_start, emo_end):
                            target_matrix[k][v] = 'E'
                            target_matrix[k][k] = 'I-E'
                    target_matrix[emo_start][emo_start] = 'B-E'

                    for k in range(cau_start, cau_end):
                        for v in range(emo_start, emo_end):
                            target_matrix[k][v] = plrt
                            target_matrix[v][k] = plrt

                    # 情绪片段序列
                    for k in range(emo_start,emo_end):
                        emotion_label[k] = 'I-E'
                        emotion_label_in_mtrx[k] = 'I-ES'
                    emotion_label[emo_start] = 'B-E'
                    emotion_label_in_mtrx[emo_start] = 'B-ES'
                    # 原因片段序列
                    for k in range(cau_start, cau_end):
                        cause_label[k] = 'I-C'
                        cause_label_in_mtrx[k] = 'I-CS'
                    cause_label[cau_start] = 'B-C'
                    cause_label_in_mtrx[cau_start] = 'B-CS'

            # emotion_label_in_mtrx = copy.deepcopy(emotion_label)
            # cause_label_in_mtrx = copy.deepcopy(cause_label)
            origin_len = len(target_matrix)
            new_len = origin_len + 9    #9是O，C，E和六个极性的长度
            cause_elements2insert = ['O','C','O','O','O','O','O','O','O']
            emotion_elements2insert = ['O','O','E','O','O','O','O','O','O']

            for single_line_index in range(len(target_matrix)): #每一行前面都要加9个标签
                target_matrix[single_line_index][0:0] = ['O','O','O','O','O','O']  #六个极性对应的标签
                target_matrix[single_line_index][0:0] = [emotion_label_in_mtrx[single_line_index]] # 标签E对应的标签
                target_matrix[single_line_index][0:0] = [cause_label_in_mtrx[single_line_index]] #  C对应的标签
                target_matrix[single_line_index][0:0] = ['O'] #O对应的标签
            six_label2insert = [['O'] * new_len for _ in range(6)]
            plrt_list = ['O', 'O', 'O', 'O', 'O', 'O']
            plrt_dict = {'anger':0,'disgust':1,'fear':2,'happiness':3,'sadness':4,'surprise':5}
            for pol in polar_label:
                plrt_list[plrt_dict[pol]] = pol
            for idx in range(len(six_label2insert)):
                six_label2insert[idx][idx+3] = plrt_list[idx]
            O_label2insert = [['O'] * new_len]

            cause_label_in_mtrx[0:0] = cause_elements2insert
            emotion_label_in_mtrx[0:0] = emotion_elements2insert
            target_matrix[0:0] = six_label2insert
            target_matrix[0:0] = [emotion_label_in_mtrx]
            target_matrix[0:0] = [cause_label_in_mtrx]
            target_matrix[0:0] = O_label2insert

            for plt_idx in range(3,9):
                if target_matrix[plt_idx][plt_idx] != 'O':
                    for text_idx in range(9,len(target_matrix)):
                        if target_matrix[text_idx][text_idx] != 'O':
                            if plt_idx == 3:
                                target_matrix[plt_idx][text_idx] = 'anger'
                                target_matrix[text_idx][plt_idx] = 'anger'
                            elif plt_idx == 4:
                                target_matrix[plt_idx][text_idx] = 'disgust'
                                target_matrix[text_idx][plt_idx] = 'disgust'
                            elif plt_idx == 5:
                                target_matrix[plt_idx][text_idx] = 'fear'
                                target_matrix[text_idx][plt_idx] = 'fear'
                            elif plt_idx == 6:
                                target_matrix[plt_idx][text_idx] = 'happiness'
                                target_matrix[text_idx][plt_idx] = 'happiness'
                            elif plt_idx == 7:
                                target_matrix[plt_idx][text_idx] = 'sadness'
                                target_matrix[text_idx][plt_idx] = 'sadness'
                            else:
                                target_matrix[plt_idx][text_idx] = 'surprise'
                                target_matrix[text_idx][plt_idx] = 'surprise'

            example_dict['docID'] = docID
            example_dict['content_data'] = content_l
            example_dict['target_data'] = target_matrix
            example_dict['clause_len'] = clause_len
            example_dict['content_len'] = len(content_l)
            example_dict['emotion_labels'] = emotion_label
            example_dict['cause_labels'] = cause_label

            return_data.append(example_dict)

        return return_data





'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    content = pickle.load(pkl_file)
    pkl_file.close()
    return content

def get_en_target(para_text,cause):
    """
    获取原因内容
    和原因内容
    """
    # text_token=para_text.split()
    text_token = para_text
    cause_token=cause.split()

    start = -1
    end = -1
    for i in range(0, len(text_token)):
        if text_token[i:i+len(cause_token)] == cause_token:
            start = i
            end = i + len(cause_token)
            return start, end

    if start == -1 or end == -1:
        print('text_token = ',text_token)
        print('cause_token = ',cause_token)
        raise ValueError("原因不在子句中")
    return start, end


def get_ch_target(para_text,cause):
    """
    获取原因内容
    和原因内容
    """
    # print('para_text = ', para_text)
    # print('cause = ', cause)
    text_token = list(para_text)
    cause_token = list(cause)

    start = -1 
    end = -1
    for i in range(0, len(text_token)):
        if text_token[i:i+len(cause_token)] == cause_token:
            start = i
            end = i + len(cause_token)
            return start, end
            
    if start == -1 or end == -1:
        print('text_token = ',text_token)
        print('cause_token = ',cause_token)
        raise ValueError("原因不在子句中")
    return start, end
    