# encoding=utf-8
import glob
import logging
import json
import time
import copy
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything, json_to_text
from tools.common import init_logger, logger
from tools.func import loadList, saveList

from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from models.bert_for_eca import Bert2Crf, Bert2Gru
from models.albert_for_ner import AlbertCrfForNer

from processors_eca.eca_seq import InputFeatures
from processors_eca.utils_eca import EcaTokenizer
from processors_eca.eca_seq import convert_examples_to_features
from processors_eca.eca_seq import eca_processors as processors, \
    batch_generator  # ner_processors = {"cner": CnerProcessor,'cluener':CluenerProcessor, 'eca':ECAProcessor}
# from processors_eca.eca_seq import collate_fn

from metrics.eca_metrics import get_prf  # 需要重新写，还没有写

from tools.finetuning_argparse_eca import get_argparse  # 需要修改一个参数，或者重新写一个类
# from processors_eca.split_seq_data import split
import numpy as np
import os
import pickle
import itertools

args = get_argparse().parse_args()
args.model_name_or_path = os.path.join(args.root_path, 'bert_base_ch')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Gpu_num)
class OneExample(object):
        """A single training/test example for token classification."""
        def __init__(self,pair_text,labels,pred_pairs,true_pairs):
            self.pair_text = pair_text
            self.label = labels
            self.pred_pairs = pred_pairs
            self.true_pairs = true_pairs

        def __repr__(self):
            return str(self.to_json_string())

        def to_dict(self):
            """Serializes this instance to a Python dictionary."""
            output = copy.deepcopy(self.__dict__)
            return output

        def to_json_string(self):
            """Serializes this instance to a JSON string."""
            return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class features(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, true_pair, pred_pair):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.true_pair = true_pair
        self.pred_pair = pred_pair

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def get_cause_span(label_list):
  span_c = []
  for i in range(len(label_list)):
    if label_list[i] == 1:
      start = i
      end = i
      while end+1<len(label_list) and label_list[end+1] == 2:
        end += 1
      span_c.append((start-1,end-1)) #减一是因为文本中不包含[CLS]，而标签中填充了[CLS]那一位
  return span_c

def get_emo_span(label_list):
    span_e = []
    for i in range(len(label_list)):
        if label_list[i] == 3:
            start = i
            end = i
            while end+1<len(label_list) and label_list[end+1] == 4:
                end += 1
            span_e.append((start-1,end-1))
    return span_e

def get_features(example_list):
    feature_list = []
    for exam in example_list:
        for i in range(len(exam.pair_text)):
            tokenizer = EcaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
            text = exam.pair_text[i][0] + exam.pair_text[i][1]
            tokens = tokenizer.tokenize(text)
            assert len(tokens) == len(text)

            special_tokens_count = 2
            max_seq_length = 64
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]

            tokens = ['CLS'] + tokens + ['SEP']



            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * max_seq_length
            padding_length = 64 - len(input_ids)

            # pad on right
            input_ids += [0] * padding_length
            input_mask += [0] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            label_ids = exam.label[i]
            feature_exam = features(input_ids,input_mask,segment_ids,label_ids,exam.true_pairs,exam.pred_pairs[i])
            feature_list.append(feature_exam)

    return feature_list


# 获取特征


def get_pair_text(text, pair):
    emo_start = pair[0][0]
    emo_end = pair[0][1]
    cause_start = pair[1][0]
    cause_end = pair[1][1]

    emo_text = text[emo_start:emo_end + 1]
    cause_text = text[cause_start:cause_end + 1]
    return emo_text, cause_text

# def train(args, train_features, model, tokenizer):
#
#     no_decay = ["bias", "LayerNorm.weight"]
#     bert_param_optimizer = list(model.bert.named_parameters())
#
#     crf_param_optimizer = list(model.crf.named_parameters())
#     linear_param_optimizer = list(model.classifier.named_parameters())
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay': args.weight_decay, 'lr': args.learning_rate},
#         {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
#          'lr': args.learning_rate},
#
#         {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
#         {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
#          'lr': args.crf_learning_rate},
#
#         {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
#         {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
#          'lr': args.crf_learning_rate}
#     ]
#     t_total = len(train_features) // args.train_batch_size * args.num_train_epochs
#     args.warmup_steps = int(t_total * args.warmup_proportion)
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
#                                                 num_training_steps=t_total)
#     # Check if saved optimizer or scheduler states exist
#     if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
#             os.path.join(args.model_name_or_path, "scheduler.pt")):
#         # Load in optimizer and scheduler states
#         optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
#     logger.info("***** Running training *****")
#     logger.info("  Num examples = %d", len(train_features))
#     logger.info("  Num Epochs = %d", args.num_train_epochs)
#     logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
#     logger.info("  Total optimization steps = %d", t_total)
#
#     global_step = 0
#     tr_loss, logging_loss = 0.0, 0.0
#     model.zero_grad()
#     seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
#     for _ in range(int(args.num_train_epochs)):
#         pbar = ProgressBar(n_total=len(train_features) // args.train_batch_size, desc='Training')
#
#         step = 0
#            #对batch进行划分   ### 还有evaluate和pridict里面的标签文本等数据没有保存下来，



def main():

    pred_label_path = os.path.join(args.root_path, 'data4classifier\\training_predicted_labels.pkl')
    true_label_path = os.path.join(args.root_path, 'data4classifier\\training_true_labels.pkl')
    text_path = os.path.join(args.root_path, 'data4classifier\\training_text.pkl')

    # 读取数据
    pred_label = []
    true_label = []
    text = []
    with open(pred_label_path, 'rb') as f:  # 读取第一步预测出来的标签
        pred_label = pickle.load(f)
        f.close()

    with open(true_label_path, 'rb') as f:
        true_label = pickle.load(f)
        f.close()

    with open(text_path, 'rb') as f:
        text = pickle.load(f)
        f.close()

    pred_emo = []  # 获取预测片段起始位置
    pred_cause = []
    for doc in pred_label:
        pred_emo.append(get_emo_span(doc))
        pred_cause.append(get_cause_span(doc))

    true_emo = []
    true_cause = []  # 获取真实片段起始位置
    for doc in true_label:
        true_emo.append(get_emo_span(doc))
        true_cause.append(get_cause_span(doc))

    # 对预测的片段做笛卡尔积
    pred_pairs = []
    for i in range(len(pred_emo)):
        doc_pairs = []
        for pair in itertools.product(pred_emo[i], pred_cause[i]):
            doc_pairs.append(pair)
        pred_pairs.append(doc_pairs)

    # 对真实的片段做笛卡尔积
    true_pairs = []
    for i in range(len(true_label)):
        doc_pairs = []
        for pair in itertools.product(true_emo[i], true_cause[i]):
            doc_pairs.append(pair)
        true_pairs.append(doc_pairs)
    # 对每个预测的pair进行标注
    # pair_labels = []  # 严格标准的标签
    # pair_text = []  # 情绪原因片段对的文本

    all_examples = []
    for doc in range(len(pred_pairs)):
        pair_text = []
        pair_labels = []

        for pair in pred_pairs[doc]:
            if pair in true_pairs[doc]:
                pair_labels.append(1)
            else:
                pair_labels.append(0)

            emo_text, cause_text = get_pair_text(text[doc], pair)
            ptext = [emo_text, cause_text]
            pair_text.append(ptext)
        doc_example = OneExample(pair_text,pair_labels,pred_pairs[doc],true_pairs[doc])
        all_examples.append(doc_example)
    # 打乱并划分数据集   # 不必再划分，现在获取的已经是第一步的训练集所得的数据了！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    np.random.seed()
    np.random.shuffle(all_examples)
    train_num = int(len(all_examples) * 0.8)
    test_num = int(len(all_examples) * 0.1)
    train_data = all_examples[0: train_num]
    test_data = all_examples[train_num : train_num + test_num]
    dev_data = all_examples[train_num + test_num : ]


    # code for training
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.device = device

    seed_everything(args.seed)
    label_list = ['no','yes']

    args.id2label = {i: label for i, label in enumerate(label_list)}  # 获取的是字典{0: no, 1: yes}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = EcaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = Bert2Crf.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None)  # 模型
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_features = get_features(train_data)
        # global_step, tr_loss = train(args, train_features, model, tokenizer)


    print(1)
if __name__ == "__main__":
    main()
