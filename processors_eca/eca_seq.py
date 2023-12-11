""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
import numpy as np
from .utils_eca import DataProcessor
import time
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels, emo_labels, cau_labels, docid = None, data_len_c =None, text_e = None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels
        self.emotion_labels = emo_labels
        self.cause_labels = cau_labels
        self.docid = docid
        self.data_len_c = data_len_c #每个子句的长度
        self.text_e = text_e


    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids, emotion_label_ids, cause_label_ids,sub_emotion_label_ids,sub_cause_label_ids, example):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.emotion_label_ids = emotion_label_ids
        self.cause_label_ids = cause_label_ids
        self.sub_emotion_label_ids = sub_emotion_label_ids
        self.sub_cause_label_ids = sub_cause_label_ids
        self.input_len = input_len
        self.example = example

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # print('examples = ', examples[0])
    label_map = {label: i for i, label in enumerate(label_list)}
    # label_map['C'] = 0  # 去除这两个标签看看会怎样
    # label_map['E'] = 0
    label_map['O'] = 0
    label_map['B-C'] = 1  #只用C和E标签
    label_map['I-C'] = 1
    label_map['C'] = 1
    label_map['B-E'] = 2
    label_map['I-E'] = 2
    label_map['E'] = 2
    label_map['anger'] = 3
    label_map['disgust'] = 4
    label_map['fear'] = 5
    label_map['happiness'] = 6
    label_map['sadness'] = 7
    label_map['surprise'] = 8

    # label_map['B-CS'] = 1
    # label_map['I-CS'] = 2
    # label_map['B-ES'] = 1
    # label_map['I-ES'] = 2
    label_map['B-CS'] = 1
    label_map['I-CS'] = 1
    label_map['B-ES'] = 2
    label_map['I-ES'] = 2
    # label_map['B-CS'] = 9
    # label_map['I-CS'] = 10
    # label_map['B-ES'] = 11
    # label_map['I-ES'] = 12
    sub_label_map = dict()
    sub_label_map['O'] = 0
    sub_label_map['B-C'] = 1
    sub_label_map['I-C'] = 2
    sub_label_map['B-E'] = 1
    sub_label_map['I-E'] = 2
    sub_label_map['B-CS'] = 1
    sub_label_map['I-CS'] = 2
    sub_label_map['B-ES'] = 1
    sub_label_map['I-ES'] = 2

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = tokenizer.tokenize(example.text_a)
        # tokens_e = tokenizer.tokenize(example.text_e) #将情感表达进行分词
        assert len(tokens) == len(example.text_a) #判断是否有的词语在token的时候被划分成为两个词语
        # assert len(tokens_e) == len(example.text_e) #判断是否有的词语在token的时候被划分成为两个词语

        # label_ids = [label_map[x] for x in example.labels]
        label_ids = [[label_map[example.labels[i][j]] for j in range(len(example.labels))] for i in range(len(example.labels))] # 把“矩阵”形式的原始标签BIO等转换为对应的index
        emotion_labels_ids = [label_map[example.emotion_labels[i]] for i in range(len(example.emotion_labels))]
        cause_labels_ids = [label_map[example.cause_labels[i]] for i in range(len(example.cause_labels))]
        sub_emotion_labels_ids = [sub_label_map[example.emotion_labels[i]] for i in range(len(example.emotion_labels))]
        sub_cause_labels_ids = [sub_label_map[example.cause_labels[i]] for i in range(len(example.cause_labels))]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        # if len(tokens) > max_seq_length - special_tokens_count:
        #     emotion_labels_ids = emotion_labels_ids[: (max_seq_length - special_tokens_count)]
        #     cause_labels_ids = cause_labels_ids[: (max_seq_length - special_tokens_count)]
        if len(tokens) > max_seq_length - special_tokens_count-9:
            tokens = tokens[: (max_seq_length - special_tokens_count-9)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)] # 待重写 此行可保留
            for i in range(max_seq_length-special_tokens_count):
                label_ids[i] = label_ids[i][:(max_seq_length-special_tokens_count)]
            emotion_labels_ids = emotion_labels_ids[: (max_seq_length - special_tokens_count-9)]
            cause_labels_ids = cause_labels_ids[: (max_seq_length - special_tokens_count-9)]
            sub_emotion_labels_ids = sub_emotion_labels_ids[: (max_seq_length - special_tokens_count - 9)]
            sub_cause_labels_ids = sub_cause_labels_ids[: (max_seq_length - special_tokens_count - 9)]
        # if len(tokens_e) > max_seq_length - special_tokens_count:
        #     tokens_e = tokens_e[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        # label_ids += [label_map['O']]

        org_len = len(label_ids)
        for i in range(org_len):
            label_ids[i].append(label_map['O'])
        label_ids.append([label_map['O']]*(org_len+1))

        emotion_labels_ids.append(label_map['O'])
        cause_labels_ids.append(label_map['O'])
        sub_emotion_labels_ids.append(sub_label_map['O'])
        sub_cause_labels_ids.append(sub_label_map['O'])

        segment_ids = [sequence_a_segment_id] * len(tokens)

        # tokens_e += [sep_token]
        # segment_e_ids = [sequence_a_segment_id] * len(tokens_e)

        if cls_token_at_end:
            tokens += [cls_token]
            # label_ids += [label_map['O']]
            org_len = len(label_ids)
            for i in range(org_len):
                label_ids[i].append(label_map['O'])
            label_ids.append([label_map['O']]*(org_len+1))

            # emotion_labels_ids.append(label_map['O'])
            # cause_labels_ids.append(label_map['O'])

            segment_ids += [cls_token_segment_id]
            
            # tokens_e += [cls_token]
            # segment_e_ids += [cls_token_segment_id]

        else:
            tokens = ['o','cause','emotion','anger','disgust','fear','happiness','sadness','surprise'] + tokens
            # tokens = ['怒','厌','恐','喜','悲','惊'] + tokens
            tokens = [cls_token] + tokens
            # label_ids = [label_map['O']] + label_ids
            # org_len = len(label_ids)
            # for i in range(org_len):
            #     label_ids[i] = [label_map['O']] + label_ids[i]
            # label_ids = ([label_map['O']] * (org_len+1)).extend(label_ids)      # extend()方法直接对原列表进行操作，不返回值，不适用于此
            # label_ids.insert(0, [label_map['O']]*(org_len+1))       # list.insert(index,value)
            # emotion_labels_ids.insert(0, label_map['O'])
            # cause_labels_ids.insert(0, label_map['O'])
            segment_ids = [0,0,0,0,0,0,0,0,0] + segment_ids
            segment_ids = [cls_token_segment_id] + segment_ids

            # tokens_e = [cls_token] + tokens_e
            # segment_e_ids = [cls_token_segment_id] + segment_e_ids

        #对input进行填充
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids) - 9
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # label_ids = ([pad_token] * padding_length) + label_ids              # 待重写

        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            # 对标签矩阵进行填充：
            # label_ids += [pad_token] * padding_length                           # 待重写,试一试使用-1填充labels
            current_len = len(label_ids)
            # for i in range(current_len):                                          # “右方填充”：首先把现有的几行 填充至最大长度
            #     label_ids[i].extend([label_map['O']] * (max_seq_length - current_len))
            # for j in range(current_len, max_seq_length):                          # 然后再添加 max_seq_length-current_len 个行
            #     label_ids.append([label_map['O']] * max_seq_length)
            for i in range(current_len):                                          # “右方填充”：首先把现有的几行 填充至最大长度
                label_ids[i].extend([-1] * (max_seq_length - current_len-1))        # 减 7 减去的是[CLS]对应的label和六个极性词对应的label
            for j in range(current_len, max_seq_length-1):                          # 然后再添加 max_seq_length-current_len 个行
                label_ids.append([-1] * (max_seq_length-1))
            for k in range(current_len-9, max_seq_length-10):  #这次改完之后，主任务的label和辅助任务label的处理就不一致了
                emotion_labels_ids.append(-1)
                cause_labels_ids.append(-1)
                sub_emotion_labels_ids.append(-1)
                sub_cause_labels_ids.append(-1)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length - 1                                 # 断言之用，非仅判别此处代码之正误而已。别处修改代码，
                                                                                  # 牵一发而动全身，断言之合理运用，其妙如调试之断点。
        assert len(emotion_labels_ids) == max_seq_length - 10
        #对情感表达进行填充

        # input_e_ids = tokenizer.convert_tokens_to_ids(tokens_e)
        # input_e_mask = [1 if mask_padding_with_zero else 0] * len(input_e_ids)
        # input_e_len = len(input_e_ids)
        # Zero-pad up to the sequence length.
        # padding_e_length = max_seq_length - len(input_e_ids)
        # if pad_on_left:
        #     input_e_ids = ([pad_token] * padding_e_length) + input_e_ids
        #     input_e_mask = ([0 if mask_padding_with_zero else 1] * padding_e_length) + input_e_mask
        #     segment_e_ids = ([pad_token_segment_id] * padding_e_length) + segment_e_ids

        # else:
        #     input_e_ids += [pad_token] * padding_e_length
        #     input_e_mask += [0 if mask_padding_with_zero else 1] * padding_e_length
        #     segment_e_ids += [pad_token_segment_id] * padding_e_length

        # assert len(input_e_ids) == max_seq_length
        # assert len(input_e_mask) == max_seq_length
        # assert len(segment_e_ids) == max_seq_length


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                      segment_ids=segment_ids, label_ids=label_ids,emotion_label_ids=emotion_labels_ids,
                                      cause_label_ids=cause_labels_ids,sub_emotion_label_ids=sub_emotion_labels_ids,
                                        sub_cause_label_ids=sub_cause_labels_ids,example = example))
    return features


#获取批量数据并打乱
def batch_generator(features, batch_size=128, return_idx=False, use_crf = True):

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_input_mask = [f.input_mask for f in features]
    all_segment_ids = [f.segment_ids for f in features]
    all_label_ids = [f.label_ids for f in features]
    all_emotion_label_ids = [f.emotion_label_ids for f in features]
    all_cause_label_ids = [f.cause_label_ids for f in features]
    all_sub_emotion_label_ids = [f.sub_emotion_label_ids for f in features]
    all_sub_cause_label_ids = [f.sub_cause_label_ids for f in features]
    all_lens = [f.input_len for f in features]
    # dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    all_example = [f.example for f in features]

    #情感词语填充
    # all_input_e_ids = [f.input_e_ids for f in features]
    # all_input_e_mask = [f.input_e_mask for f in features]
    # all_segment_e_ids = [f.segment_e_ids for f in features]
    # all_e_lens = [f.input_e_len for f in features]


    for offset in range(0, len(features), batch_size):
    # for offset in range(0, 2*batch_size, batch_size):
    #     batch_generator_start = time.time()
        #获取x的长度
        input_mask = all_input_mask[offset:offset+batch_size] #为了计算长度
        batch_x_len = np.sum(input_mask, -1)
        max_doc_len =  max(batch_x_len) #文本的最大长度
        batch_idx=batch_x_len.argsort()[::-1]

        input_ids = np.array(all_input_ids[offset:offset+batch_size])[batch_idx]
        input_mask = np.array(all_input_mask[offset:offset+batch_size])[batch_idx]
        segment_ids = np.array(all_segment_ids[offset:offset+batch_size])[batch_idx]
        label_ids = np.array(all_label_ids[offset:offset+batch_size])[batch_idx]
        emotion_ids = np.array(all_emotion_label_ids[offset:offset+batch_size])[batch_idx]
        cause_ids = np.array(all_cause_label_ids[offset:offset+batch_size])[batch_idx]
        sub_emotion_ids = np.array(all_sub_emotion_label_ids[offset:offset + batch_size])[batch_idx]
        sub_cause_ids = np.array(all_sub_cause_label_ids[offset:offset + batch_size])[batch_idx]
        raw_labels = np.array(all_label_ids[offset:offset+batch_size])[batch_idx]
        lens = np.array(all_lens[offset:offset+batch_size])[batch_idx]
        # 情感信息
        # input_e_ids = np.array(all_input_e_ids[offset:offset+batch_size])[batch_idx]
        # input_e_mask = np.array(all_input_e_mask[offset:offset+batch_size])[batch_idx]
        # segment_e_ids = np.array(all_segment_e_ids[offset:offset+batch_size])[batch_idx]
        # lens_e = np.array(all_e_lens[offset:offset+batch_size])[batch_idx]

        batch_example = [all_example[offset:offset+batch_size][i] for i in batch_idx]
        #转换为torch类型
        # print('batch_x = ',batch_x)
        batch_input_ids = torch.from_numpy(input_ids[0:max_doc_len]).long().cuda()
        batch_input_mask = torch.from_numpy(input_mask[0:max_doc_len]).long().cuda()
        batch_segment_ids = torch.from_numpy(segment_ids[0:max_doc_len]).long().cuda()
        batch_label_ids = torch.from_numpy(label_ids[0:max_doc_len]).long().cuda()
        batch_emotion_ids = torch.from_numpy(emotion_ids[0:max_doc_len]).long().cuda()
        batch_cause_ids = torch.from_numpy(cause_ids[0:max_doc_len]).long().cuda()
        batch_sub_emotion_ids = torch.from_numpy(sub_emotion_ids[0:max_doc_len]).long().cuda()
        batch_sub_cause_ids = torch.from_numpy(sub_cause_ids[0:max_doc_len]).long().cuda()
        batch_raw_labels = torch.from_numpy(raw_labels[0:max_doc_len]).long().cuda()
        batch_lens = torch.from_numpy(lens).long().cuda()

        # batch_input_e_ids = torch.from_numpy(input_e_ids[0:max_doc_len]).long().cuda()
        # batch_input_e_mask = torch.from_numpy(input_e_mask[0:max_doc_len]).long().cuda()
        # batch_segment_e_ids = torch.from_numpy(segment_e_ids[0:max_doc_len]).long().cuda()
        # batch_e_lens = torch.from_numpy(lens_e).long().cuda()

        if len(batch_label_ids.size())==2 and not use_crf:
            batch_label_ids = torch.nn.utils.rnn.pack_padded_sequence(batch_label_ids, batch_lens.to('cpu'), batch_first = True)

        # batch_generator_end = time.time()
        # print("batch generator time consuming: ", batch_generator_end - batch_generator_start)

        if return_idx: #in testing, need to sort back.
            yield (batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens,  batch_label_ids, batch_emotion_ids, batch_cause_ids,batch_sub_emotion_ids,batch_sub_cause_ids, batch_idx, batch_raw_labels, batch_example)
        else:
            yield (batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens,  batch_label_ids, batch_emotion_ids, batch_cause_ids,batch_sub_emotion_ids,batch_sub_cause_ids, batch_raw_labels, batch_example)

class ECA_en_Processor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_en_pkl(data_path = os.path.join(data_dir, "eca_train.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_en_pkl(data_path = os.path.join(data_dir, "eca_dev.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_en_pkl(data_path = os.path.join(data_dir, "eca_test.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["O", "C-B", "C-I","E-B","E-I"]
        return ['O', 'B-C', 'I-C', 'C', 'B-E', 'I-E', 'E', 'anger', 'disgust', 'fear', 'happiness', 'sadness',
                'surprise']
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['content_data']
            labels = line['target_data'] #BIO 当前文本的标签 list列表
            docid = line['docID']
            emo_tokens = line['emo_data']
            # emotion_index = line['emotion_index']
            data_len_c = line['clause_len']
            # ec_index = line['ec_index']
            # BIOS
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels, docid = docid, data_len_c= data_len_c, text_e = emo_tokens))
        return examples


class ECA_ch_Processor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ch_pkl(data_path = os.path.join(data_dir, "eca_train.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ch_pkl(data_path = os.path.join(data_dir, "eca_dev.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ch_pkl(data_path = os.path.join(data_dir, "eca_test.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["O", "A-C-B", "A-C-I","A-E-B","A-E-I","D-C-B","D-C-I","D-E-B","D-E-I","F-C-B","F-C-I","F-E-B","F-E-I","H-C-B","H-C-I","H-E-B","H-E-I","SD-C-B","SD-C-I","SD-E-B","SD-E-I","SP-C-B","SP-C-I","SP-E-B","SP-E-I"]
        # return ['O', 'B-C', 'I-C', 'C', 'B-E', 'I-E', 'E', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        return ['O', 'B-C', 'I-C', 'C', 'B-E', 'I-E', 'E', 'anger', 'disgust', 'fear', 'happiness', 'sadness',
                'surprise']
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['content_data']
            labels = line['target_data'] #BIO 当前文本的标签 list列表
            docid = line['docID']
            emo_tokens = line['emo_data']
            # emotion_index = line['emotion_index']
            data_len_c = line['clause_len']
            # ec_index = line['ec_index']
            emotion_labels = line['emotion_labels']
            cause_labels = line['cause_labels']
            # BIOS
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels,emo_labels=emotion_labels,cau_labels=cause_labels, docid = docid, data_len_c= data_len_c, text_e = emo_tokens))
        return examples

class ECA_merge_ch_Processor(DataProcessor):
    """processor for the merged chinese dataset"""
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_merge_ch_pkl(data_path = os.path.join(data_dir, "eca_train.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_merge_ch_pkl(data_path = os.path.join(data_dir, "eca_dev.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_merge_ch_pkl(data_path=os.path.join(data_dir, "eca_test.pkl"), save_csv_path=os.path.join(data_dir, "ecatext_test.csv")),"test")

    def get_labels(self):
        return ['O', 'B-C', 'I-C', 'C', 'B-E', 'I-E', 'E', 'anger', 'disgust', 'fear', 'happiness', 'sadness',
                'surprise','B-ES','I-ES','B-CS','I-CS']
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['content_data']
            labels = line['target_data']  # BIO 当前文本的标签 list列表
            emotion_labels = line['emotion_labels']
            cause_labels = line['cause_labels']
            docid = line['docID']
            # emo_tokens = line['emo_data']
            # emotion_index = line['emotion_index']
            data_len_c = line['clause_len']
            # ec_index = line['ec_index']
            # BIOS
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels,emo_labels=emotion_labels,cau_labels=cause_labels, docid=docid, data_len_c=data_len_c))
        return examples
eca_processors = {
    'en':ECA_en_Processor,
    'ch':ECA_ch_Processor,
    'merge_ch':ECA_merge_ch_Processor,
    # 'sti':ECA_sti_Processor
}
