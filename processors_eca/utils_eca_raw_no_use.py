import csv
import json
import torch
from models.transformers import BertTokenizer
import pickle
import codecs
import re

class EcaTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False):
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

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})
        return lines

    @classmethod
    def _read_json(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label',None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key,value in label_entities.items():
                        for sub_name,sub_index in value.items():
                            for start_index,end_index in sub_index:
                                assert  ''.join(words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-'+key
                                else:
                                    labels[start_index] = 'B-'+key
                                    labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
                lines.append({"words": words, "labels": labels})
        return lines
    

    # @classmethod
    # def _read_pkl(self,input_file):
    #     lines = []
    #     with open(input_file,'r') as f:
    #         for line in f:
    #             line = json.loads(line.strip())
    #             text = line['text']
    #             label_entities = line.get('label',None)
    #             words = list(text)
    #             labels = ['O'] * len(words)
    #             if label_entities is not None:
    #                 for key,value in label_entities.items():
    #                     for sub_name,sub_index in value.items():
    #                         for start_index,end_index in sub_index:
    #                             assert  ''.join(words[start_index:end_index+1]) == sub_name
    #                             if start_index == end_index:
    #                                 labels[start_index] = 'S-'+key
    #                             else:
    #                                 labels[start_index] = 'B-'+key
    #                                 labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
    #             lines.append({"words": words, "labels": labels})
    #     return lines

    def _read_pkl(self,data_path, save_csv_path):
        """
        将数据读取和写入 获取pkl文件和csv文件
        """
        data = loadList(data_path)
        outputFile1 = codecs.open(save_csv_path, 'w','utf-8') #将文本写入到csv文件
        out_data= [] #是一个list列表，每一个list都存入一个字典
        for index, item in enumerate(data): #对于每一个文本
            para_data = dict() #将每一个文本写成一个字典的形式
            write_str = '' #每一个文本吸入的内容暂时存在这个字符串里
            content_Data = ''#每一个文本的内容
            emotion_Data = ''#情感内容
            target_Data = ''#标签内容
            data_len = []#每一个子句的长度
            docID = item[0]['docID'] #文本的ID号

            target_d_l = [] #每一个文本的标签list
            content_d_l = [] #每一个文本的content list

            clause_info = item[4] #所有的子句信息
            key_loc= int(item[2][0]['keyloc']) #情感子句所在的索引
            emotion_word = item[2][0]['keyword'] #情感词语的内容
        
            ec_index = [0] * len(clause_info)
            ec_index[key_loc] = 1 #情感子句处为1，其余位置为0

            emotion_Data = clause_info[key_loc]['content'].strip()#情感子句的内容
            ew_index_span = re.search(re.escape(emotion_word), emotion_Data).span()
            emotion_list = list(emotion_Data)
            # print('emotion_list = ', emotion_list)
            ew_index_l = []
            # ec_index_l = [] #长度仍然为文本中词语的总的个数，比如相同句子中的位置
            for indexc, itemc in enumerate(clause_info):
                clause = itemc['content']
                cause = itemc['cause_content']

                clause_l = list(clause)
                target_l = ['O'] * len(clause_l)
                data_len.append(len(clause_l))

                ew_index = [0] * len(clause_l)
                if key_loc == indexc:
                    start, end = ew_index_span[0], ew_index_span[1]
                    for i in range(start, end):
                        ew_index[i] = 1
                ew_index_l.extend(ew_index) #将每个子句进行拼接，目的是在情感词语的位置赋值为1 其余的位置赋值为0

                if cause != '':
                    target_l = ['O'] * len(clause_l)
                    target_l = get_target(target_l, clause, cause)
                
                content_Data += clause
                target_d_l.extend(target_l)
                content_d_l.extend(clause_l)
            
            para_data['docID'] = docID
            para_data['emotion_Data'] = emotion_list
            para_data['content_Data'] = content_d_l
            para_data['target_Data'] = target_d_l
            para_data['emotion_index'] = ew_index_l
            para_data['data_len'] = data_len
            para_data['ec_index'] = ec_index
            out_data.append(para_data)

            #将每一个文本内容写入csv文件
            ss = ''
            for x, y in zip(content_d_l, target_d_l):
                ss += str(x) + '#' + str(y) + ' '

            write_str += str(docID) + '\n' + ss.strip() + '\n' + emotion_Data + '\n\n'
            outputFile1.write(write_str)
        outputFile1.close()
        # saveList(out_data, save_data_path)
        return out_data
    


'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    content = pickle.load(pkl_file)
    pkl_file.close()
    return content

"""
获取标签 B I O
"""
def get_target(target_l, clause, cause):
    """
    获取数据的BIO 获取情感原因的BIO
    """
    cause_span = re.search(re.escape(cause), clause).span()
    start = cause_span[0]
    end = cause_span[1]
    target_l[start] = 'B'
    if start != end -1:
       for  i in range(start + 1, end):
           target_l[i] = 'I'
    return target_l

def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup='bio'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    else:
        return get_entity_bios(seq,id2label)

def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S
