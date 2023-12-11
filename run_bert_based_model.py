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
from models.bert_for_eca import Bert2Crf, Bert2Gru, Bert2Softmax
from models.albert_for_ner import AlbertCrfForNer

from processors_eca.utils_eca import EcaTokenizer
from processors_eca.eca_seq import convert_examples_to_features
from processors_eca.eca_seq import eca_processors as processors, \
    batch_generator  # ner_processors = {"cner": CnerProcessor,'cluener':CluenerProcessor, 'eca':ECAProcessor}
# from processors_eca.eca_seq import collate_fn

# from metrics.eca_metrics import get_prf  # 需要重新写，还没有写
from metrics.eca_metrics import get_accuracy, get_accuracy_2, get_accuracy_3
# from metrics.eca_metrics import get_cause_span,get_emo_span
from tools.finetuning_argparse_eca import get_argparse  # 需要修改一个参数，或者重新写一个类
# from processors_eca.split_seq_data import split
import numpy as np
import pandas as pd
import os

args = get_argparse().parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Gpu_num)

if args.model_encdec == "bert2softmax":
    MODEL_CLASSES = {
        'bert': (BertConfig, Bert2Softmax, EcaTokenizer)
    }

if args.data_type == 'ch' or args.data_type == 'merge_ch':
    args.model_name_or_path = os.path.join(args.root_path, 'bert_base_ch')
elif args.data_type == 'en' or args.data_type == 'sti':
    args.model_name_or_path = os.path.join(args.root_path, 'bert_base_en')


def train(args, train_features, model, tokenizer, use_crf):
    """ Train the model """
    # args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    # 这里用的是随机数据，下面evalue 的时候用的是顺序数据集
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # if args.model_encdec == 'bert2gru' or args.model_encdec == 'bert2gruAtt':
    #     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
    #                               collate_fn=collate_fn)#每个数据集划分的batch的个数
    # elif args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2crfAtt':
    #     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
    #                               collate_fn=collate_fn)#每个数据集划分的batch的个数

    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())

    if args.model_encdec == 'bert2softmax':
        tuning_param_optimizer = [parameter for parameter in model.named_parameters() if str(parameter[0]).split('.')[0] != 'bert']
        # linear_param_optimizer = list(model.classifier.named_parameters())
        # linear_param_pre_concat_1 = list(model.pre_concat_fc1.named_parameters())
        # linear_param_pre_concat_2 = list(model.pre_concat_fc2.named_parameters())
        # linear_param_polarity_fc = list(model.polarity_fc.named_parameters())
        # linear_param_emotion_fc = list(model.classifier_emotion.named_parameters())
        # lienar_param_cause_fc = list(model.classifier_cause.named_parameters())
        # linear_param_cause_fc1 = list(model.fc_cause.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},

            {'params': [p for n, p in tuning_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in tuning_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate},

        ]

    t_total = len(train_features) // args.train_batch_size * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    total_step = 0
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_features) // args.train_batch_size, desc='Training')
        # 修改lixiangju
        step = 0

        for batch in batch_generator(features=train_features, batch_size=args.train_batch_size, use_crf=use_crf):
            # batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens, batch_label_ids, batch_raw_labels, batch_example = batch
            model.train()

            if args.model_encdec == 'bert2softmax':
                batch_inputs = tuple(t.to(args.device) for t in batch[0:9])
                inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],
                          "token_type_ids": batch_inputs[2], "labels": batch_inputs[4],
                          "emotion_labels":batch_inputs[5], "cause_labels":batch_inputs[6],
                          "sub_emotion_labels":batch_inputs[7],"sub_cause_labels":batch_inputs[8],"testing": False}
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()

            pbar(step, {'loss': loss.item()})
            step += 1
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # optimizer.step()  # 根据系统warning， 将此行代码与下行代码位置互换
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     # Log metrics
                #     print(" ")
                #
                #     if args.local_rank == -1 and not args.ten_fold:
                #         # Only evaluate when single GPU otherwise metrics may not average well
                #         results = evaluate(args=args, model=model, tokenizer=tokenizer, use_crf=use_crf)


                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = (
                #         model.module if hasattr(model, "module") else model
                #     )  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
                #     logger.info("Saving model checkpoint to %s", output_dir)
                #     tokenizer.save_vocabulary(output_dir)
                #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                #     logger.info("Saving optimizer and scheduler states to %s", output_dir)
        print(" ")

        # train_history.append(loss)
        np.random.seed()
        np.random.shuffle(train_features)
        logger.info("\n")
        # if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", use_crf=False):
    # metric = get_prf(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_features = load_and_cache_examples(args, args.data_type, tokenizer, data_type='dev')
    processor = processors[args.data_type]()

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    # nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module

    pre_labels, tru_labels, eval_examples = [], [], []  # 统计整个eval数据的 标签
    # pbar = ProgressBar(n_total=len(eval_features)//args.train_batch_size, desc='Training')
    step = 0
    for batch in batch_generator(features=eval_features, batch_size=args.train_batch_size, use_crf=use_crf):

        batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens, batch_label_ids, batch_raw_labels, batch_example = batch
        model.eval()
        if args.model_encdec == 'bert2crfAtt' or args.model_encdec == 'bert2gruAtt':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:7])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],
                      "token_type_ids": batch_inputs[2], "labels": None,
                      "testing": True}

        elif args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2gru':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],
                      "token_type_ids": batch_inputs[2], "labels": None, "testing": True}

        elif args.model_encdec == 'bert2softmax':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],
                      "token_type_ids": batch_inputs[2], "labels": None, "emotion_labels":None, "cause_labels":None, "testing": True}
        outputs = model(**inputs)
        eval_examples.extend(batch_example)
        # out_label_ids = batch[5].tolist()  # 真实标签
        out_label_ids = batch[5].cpu().numpy()
        if use_crf:
            logits = outputs[0]
            # crf attention mask
            crf_attention_mask = inputs['attention_mask'].expand(inputs['attention_mask'].shape[1],inputs['attention_mask'].shape[0],inputs['attention_mask'].shape[1])\
            .transpose(0, 1)\
            .reshape(inputs['attention_mask'].shape[0], inputs['attention_mask'].shape[1]**2)
            tags = model.crf.decode(logits, crf_attention_mask)
            # eval_loss += tmp_eval_loss.item()
            # nb_eval_steps += 1
            # input_lens = batch_inputs[3].cpu().numpy().tolist()
            # tags = tags.squeeze(0).cpu().numpy().tolist()
            tags = tags.squeeze(0).cpu().numpy().reshape(out_label_ids.shape[0], out_label_ids.shape[1], out_label_ids.shape[2])
            # for len_doc, cu_tags, cu_trus in zip(batch_lens, tags, out_label_ids):
            #     pre_labels.append(cu_tags[0: len_doc])
            #     tru_labels.append
            for single_index in range(tags.shape[0]):
                len_doc = batch_lens[single_index]
                pre_labels.append(tags[single_index, 0:len_doc, 0:len_doc])           # list里面装的是一个个numpy array, 每个numpy array是一个截取出来的方块儿
                tru_labels.append(out_label_ids[single_index, 0:len_doc, 0:len_doc])

        else:
            tags = outputs[0].detach().cpu().numpy().argmax(-1)
            # tags = tags.tolist()
            # # out_label_ids = batch[5].tolist() #真实标签
            # # pre_labels =[list(p[i]) for i in range(p.shape[0])]
            # for len_doc, cu_tags, cu_trus in zip(batch_lens, tags, out_label_ids):
            #     pre_labels.append(cu_tags[0: len_doc])
            #     tru_labels.append(cu_trus[0: len_doc])
            tags = tags.reshape(out_label_ids.shape[0], out_label_ids.shape[1], out_label_ids.shape[2])
            for single_index in range(tags.shape[0]):
                len_doc = batch_lens[single_index]
                pre_labels.append(tags[single_index, 0:len_doc, 0:len_doc])           # list里面装的是一个个numpy array, 每个numpy array是一个截取出来的方块儿
                tru_labels.append(out_label_ids[single_index, 0:len_doc, 0:len_doc])

        step += 1
        pbar(step)


    logger.info("\n")
    # eval_loss = eval_loss / nb_eval_steps
    # eval_info, entity_info = metric.result()
    results = get_accuracy(pre_labels, tru_labels, eval_examples)  # 此三者皆为list，前两者里面装的是numpy array，eval_examples装的是example,example中的labels是双层嵌套list
    # results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    print('\n results = ', results, "\n")
    return results


def predict(args, model, tokenizer, prefix="", use_crf=False):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    print('args.data_type test = ', args.data_type)
    test_features = load_and_cache_examples(args, args.data_type, tokenizer, data_type='test')
    processor = processors[args.data_type]()

    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_features))
    logger.info("  Batch size = %d", 1)
    # results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_features), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module

    pre_labels, tru_labels, test_examples = [], [], []  # 统计整个eval数据的 标签
    pre_labels_emotion, tru_labels_emotion = [], []
    pre_labels_cause, tru_labels_cause = [], []
    pre_labels_emotion_sub, pre_labels_cause_sub = [], []
    tru_labels_emotion_sub, tru_labels_cause_sub = [], []
    pre_labels_l, tru_labels_l = [], []
    step = 0
    for batch in batch_generator(features=test_features, batch_size=args.train_batch_size, use_crf=use_crf):
        # batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens, batch_label_ids, batch_raw_labels, batch_example = batch
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens, batch_label_ids, batch_emotion_ids,batch_cause_ids,sub_batch_emotion_ids,sub_batch_cause_ids, batch_raw_labels, batch_example = batch
        model.eval()
        if args.model_encdec == 'bert2crfAtt' or args.model_encdec == 'bert2gruAtt':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],
                      "token_type_ids": batch_inputs[2], "labels": None,
                      "testing": True}

        elif args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2gru':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],
                      "token_type_ids": batch_inputs[2], "labels": None, "testing": True}

        elif args.model_encdec == 'bert2softmax':
            # batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
            # inputs = {"inputs_ids": batch_inputs[0], "attention_mask": batch_inputs[1],
            #           "token_type_ids": batch_inputs[2], "labels": None, "testing": True}
            batch_inputs = tuple(t.to(args.device) for t in batch[0:9])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],
                      "token_type_ids": batch_inputs[2], "labels": None, "emotion_labels":None, "cause_labels":None, "testing": True}


        outputs = model(**inputs)
        test_examples.extend(batch_example)
        out_label_ids = batch[4].cpu().numpy()  # 真实标签
        out_label_ids_list = batch[4].tolist()  # 后续带_list和_l的，是为了方便json to text
        emotion_out_label_ids = batch[5].cpu().numpy()
        cause_out_label_ids = batch[6].cpu().numpy()
        sub_batch_emotion_ids = batch[7].cpu().numpy()
        sub_batch_cause_ids = batch[8].cpu().numpy()
        if use_crf:
            logits = outputs[0]

            crf_attention_mask = inputs['attention_mask'].expand(inputs['attention_mask'].shape[1],
                                                                 inputs['attention_mask'].shape[0],
                                                                 inputs['attention_mask'].shape[1]) \
                .transpose(0, 1) \
                .reshape(inputs['attention_mask'].shape[0], inputs['attention_mask'].shape[1] ** 2)

            tags = model.crf.decode(logits, crf_attention_mask)
            # eval_loss += tmp_eval_loss.item()
            # nb_eval_steps += 1
            # input_lens = batch_inputs[3].cpu().numpy().tolist()

            tags_l = tags.squeeze(0).cpu().numpy().tolist()
            for len_doc, cu_tags, cu_trus in zip(batch_lens, tags_l, out_label_ids_list):
                pre_labels_l.append(cu_tags[0: len_doc])
                tru_labels_l.append(cu_trus[0: len_doc])

            tags = tags.squeeze(0).cpu().numpy().reshape(out_label_ids.shape[0], out_label_ids.shape[1],out_label_ids.shape[2])
            for single_index in range(tags.shape[0]):
                len_doc = batch_lens[single_index]
                pre_labels.append(
                    tags[single_index, 0:len_doc, 0:len_doc])  # list里面装的是一个个numpy array, 每个numpy array是一个截取出来的方块儿
                tru_labels.append(out_label_ids[single_index, 0:len_doc, 0:len_doc])

        else:
            tags = outputs[0][:,9:,9:,:].detach().cpu().numpy().argmax(-1)
            # tags_emotion = outputs[1].detach().cpu().numpy().argmax(-1)
            # tags_cause = outputs[2].detach().cpu().numpy().argmax(-1)
            tags_emotion = outputs[0][:,2,9:,:].detach().cpu().numpy().argmax(-1)
            tags_cause = outputs[0][:,1,9:,:].detach().cpu().numpy().argmax(-1)
            tags_emotion_sub = outputs[1].detach().cpu().numpy().argmax(-1)
            tags_cause_sub = outputs[2].detach().cpu().numpy().argmax(-1)
            out_label_ids = out_label_ids[:,9:,9:]  #这个真实标签也要从下标为9的地方开始
            # tags = tags.tolist()
            # # out_label_ids = batch[5].tolist() #真实标签
            # # pre_labels =[list(p[i]) for i in range(p.shape[0])]
            #
            # for single_index in range(tags.shape[0]):
            #     len_doc = batch_lens[single_index]
            #     pre_labels.append(tags[single_index, 0:len_doc, 0:len_doc])           # list里面装的是一个个numpy array, 每个numpy array是一个截取出来的方块儿
            #     tru_labels.append(out_label_ids[single_index, 0:len_doc, 0:len_doc])
            # for len_doc, cu_tags, cu_trus in zip(batch_lens, tags, out_label_ids):
            #     pre_labels.append(cu_tags[0: len_doc])
            #     tru_labels.append(cu_trus[0: len_doc])
            tags = tags.reshape(tags.shape[0], tags.shape[1], tags.shape[2])
            tags_emotion = tags_emotion.reshape(tags_emotion.shape[0], tags_emotion.shape[1])
            tags_cause = tags_cause.reshape(tags_cause.shape[0],tags_cause.shape[1])
            tags_emotion_sub = tags_emotion_sub.reshape(tags_emotion_sub.shape[0],tags_emotion_sub.shape[1])
            tags_cause_sub = tags_cause_sub.reshape(tags_cause_sub.shape[0],tags_cause_sub.shape[1])
            for single_index in range(tags.shape[0]):
                len_doc = batch_lens[single_index]
                pre_labels.append(
                    tags[single_index, 0:len_doc, 0:len_doc])  # list里面装的是一个个numpy array, 每个numpy array是一个截取出来的方块儿
                tru_labels.append(out_label_ids[single_index, 0:len_doc, 0:len_doc])
            for single_index in range(tags_emotion.shape[0]):
                len_doc = batch_lens[single_index]
                pre_labels_emotion.append(
                    tags_emotion[single_index, 0:len_doc])
                tru_labels_emotion.append(emotion_out_label_ids[single_index, 0:len_doc])
                pre_labels_cause.append(
                    tags_cause[single_index, 0:len_doc]
                )
                pre_labels_emotion_sub.append(
                    tags_emotion_sub[single_index, 0:len_doc]
                )
                pre_labels_cause_sub.append(
                    tags_cause_sub[single_index, 0:len_doc]
                )
                tru_labels_cause.append(cause_out_label_ids[single_index, 0:len_doc])
                tru_labels_emotion_sub.append(sub_batch_emotion_ids[single_index, 0:len_doc])
                tru_labels_cause_sub.append(sub_batch_cause_ids[single_index, 0:len_doc])
        step += 1
        pbar(step)

    logger.info("\n")
    # eval_loss = eval_loss / nb_eval_steps
    # eval_info, entity_info = metric.result()
    results = get_accuracy_3(pre_labels, tru_labels, test_examples,pre_labels_emotion,pre_labels_emotion_sub,tru_labels_emotion,tru_labels_emotion_sub,pre_labels_cause,pre_labels_cause_sub,tru_labels_cause,tru_labels_cause_sub)

    # save pre_labels and tru_labels
    if not os.path.exists("./save_pre_tru"):
        os.mkdir("./save_pre_tru")
    saveList(pre_labels, "./save_pre_tru/predicted_labels.pkl")
    saveList(tru_labels, "./save_pre_tru/true_labels.pkl")

    # results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    print('results = ', results)

    output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
    test_submit = []
    for pre, tru, exam in zip(pre_labels_l, tru_labels_l, test_examples):
        json_d = {}
        json_d['docid'] = exam.docid
        json_d['words'] = exam.text_a
        # json_d['tru_cause'] = get_content(exam.text_a, tru)
        # json_d['pre_cause'] = get_content(exam.text_a, pre)
        json_d['tag_seq'] = " ".join([args.id2label[x] for x in pre])
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)
    return results


def get_content(text, label):
    """
    text: list[str]
    tru: list[int]
    """
    content = []
    for index, item in enumerate(label[1:-1]):
        if item != 0:
            content.append(text[index])
    return content


# 加载数据，并转换为tensor类型，还进行tokenizer
def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()  # [B I O]
    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir)  # 获取数据，并且 增加了将数据存储为csv文件
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                else args.eval_max_seq_length,  # 需要测试一下两部分数据的最大长度是什么
                                            cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )
    return features


def train_model():
    # 允许重写，也就是每次允许的时候，不用担心里面有文件
    # if os.path.exists(args.output_dir) and os.listdir( #如果存在文件不为空，并且不能覆盖重写，那么就返回错误，在这里程序设置为可以重写
    #         args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #             args.output_dir))

    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1

    if args.model_encdec == 'bert2gru' or args.model_encdec == 'bert2gruAtt' or args.model_encdec == 'bert2gruCoAtt' or args.model_encdec == 'bert2softmax':
        use_crf = False
    elif args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2crfAtt' or args.model_encdec == 'bert2crfCoAtt':
        use_crf = True

    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), )
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.data_type = args.data_type.lower()  # 将任务名称设置为小写
    if args.data_type not in processors:  # 如果任务名称不在列表中，那么返回错误,列表为 ner_processors = {"cner": CnerProcessor,'cluener':CluenerProcessor, 'eca':ECAProcessor}
        raise ValueError("Task not found: %s" % (args.data_type))
    processor = processors[args.data_type]()
    label_list = processor.get_labels()  # 这里是获取的标签列表，本任务：【B I O】 现在是【C-B C-I E-B E-I O】

    args.id2label = {i: label for i, label in enumerate(label_list)}  # 获取的是字典{0: O, 1: C-B, 2: C-I, 3: E-B, 4: E-I}
    args.label2id = {label: i for i, label in enumerate(label_list)}  # 获取的是字典{O:0, C-B:1, C-I: 2, E-B: 3, E-I: 4}
    num_labels = len(label_list)  # 标签的个数

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[
        args.model_type]  # BertConfig, BertCrfForNer, EcaTokenizer
    # BertConfig.from_pretrained
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None)  # 模型
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_features = load_and_cache_examples(args, args.data_type, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_features, model, tokenizer, use_crf=use_crf)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0] and not args.ten_fold:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args=args, model=model, tokenizer=tokenizer, prefix=prefix, use_crf=use_crf)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            metrics = predict(args, model, tokenizer, prefix=prefix, use_crf=use_crf)
    return metrics


def main():
    if not os.path.exists(args.results_dir):  # 如果不存在进行创建
        os.mkdir(args.results_dir)

    list_result = []
    # print(args.split_times)
    out_current_path = copy.deepcopy(args.output_dir)

    for i in range(args.split_times):
        print("*****************************split_times:{}*******************".format(i))
        args.output_dir = args.output_dir + '{}_{}_{}_{}'.format(args.data_type, args.model_encdec, args.Gpu_num,
                                                                 i)  # 输出模型文件的位置
        if os.path.exists(args.output_dir):  # 是否存在输出文件，如果不存在进行创建
            shutil.rmtree(args.output_dir)
            print('删除文件夹：', args.output_dir)
        os.mkdir(args.output_dir)
        # 观测数据文件中是否包含数据
        data_current_path = copy.deepcopy(args.data_dir)
        args.data_dir = args.data_dir + '{}_{}_{}_{}'.format(args.data_type, args.model_encdec, args.Gpu_num, i)
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
        os.mkdir(args.data_dir)
        # j加载数据并划分

        # 如果使用合并之后的数据集：


        # if args.data_type == 'ch':
        data_set = loadList(os.path.join(args.dataset_eca_raw, 'eca_{}.pkl'.format(args.data_type)))

        # if args.data_type == 'en':
        # data_set = loadList(os.path.join(args.dataset_eca_raw, 'eca_en_data.pkl'))
        print('data_set = ', len(data_set))
        # 每次划分之前打乱数据集进行数据划分
        np.random.seed()
        np.random.shuffle(data_set)
        train_num = int(len(data_set) * 0.8)
        test_num = int(len(data_set) * 0.1)

        train_data = data_set[0: train_num]
        test_data = data_set[train_num: train_num + test_num]
        dev_data = data_set[train_num + test_num:]

        # 保存数据到对应的文件夹中
        saveList(train_data, os.path.join(args.data_dir, 'eca_train.pkl'))
        saveList(test_data, os.path.join(args.data_dir, 'eca_test.pkl'))
        saveList(dev_data, os.path.join(args.data_dir, 'eca_dev.pkl'))

        # 获取一次划分之后测试集合上的结果
        metrics = train_model()
        list_result.append(metrics)
        print('\n完成当前划分的实验结果：bert_crf_results_{} = {}\n'.format(args.data_type, metrics))

        # 删除当前存储数据文件
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
            print('删除文件夹：', args.data_dir)

        # 将存储文件基本目录更新到原来样子
        args.output_dir = out_current_path
        args.data_dir = data_current_path

    print('完成所有划分，数据{}结果如下：\n'.format(args.data_type))
    result_v = ''
    for index, result in enumerate(list_result):
        strr = ''
        for key, value in result.items():
            strr += key + '\t'
            result_v += str(value) + '\t'
        result_v += '\n'
        print(strr)
        print(result_v)

    results_path = os.path.join(args.results_dir,
                                '{}_{}_{}_{}_results.csv'.format(args.model_encdec, args.data_type, args.att_type,
                                                                 args.Gpu_num))
    f = open(results_path, 'w')
    f.write(strr)
    f.write('\n')
    f.write(result_v)
    f.close()

def main2():
    if not os.path.exists(args.results_dir):  # 如果不存在进行创建
        os.mkdir(args.results_dir)

    list_result = []
    out_current_path = copy.deepcopy(args.output_dir)


    data_set = loadList(os.path.join(args.dataset_eca_raw, 'eca_{}.pkl'.format(args.data_type)))
    print('data_set = ', len(data_set))
    np.random.seed()
    np.random.shuffle(data_set)

    one_fold_len = int(len(data_set) / 10)
    ten_fold = []
    for i in range(9):
        ten_fold.append(data_set[i * one_fold_len:(i + 1) * one_fold_len])
    ten_fold.append(data_set[one_fold_len * 9:])

    for j in range(10): # 每一折
        print("*****************************fold:{}*******************".format(j))

        args.output_dir = args.output_dir + '{}_{}_{}_{}'.format(args.data_type, args.model_encdec, args.Gpu_num,
                                                                 j)  # 输出模型文件的位置
        if os.path.exists(args.output_dir):  # 是否存在输出文件，如果不存在进行创建
            shutil.rmtree(args.output_dir)
            print('删除文件夹：', args.output_dir)
        os.mkdir(args.output_dir)
        # 观测数据文件中是否包含数据
        data_current_path = copy.deepcopy(args.data_dir)
        args.data_dir = args.data_dir + '{}_{}_{}_{}'.format(args.data_type, args.model_encdec, args.Gpu_num, j)
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
        os.mkdir(args.data_dir)

        test_set = ten_fold[j]
        train_set1 = ten_fold[0:j]
        train_set2 = ten_fold[j + 1:]
        train_set = []
        for k in range(len(train_set1)):
            train_set.extend(train_set1[k])
        for k in range(len(train_set2)):
            train_set.extend(train_set2[k])

        # 保存数据到对应的文件夹中
        saveList(train_set, os.path.join(args.data_dir, 'eca_train.pkl'))
        saveList(test_set, os.path.join(args.data_dir, 'eca_test.pkl'))

        # 获取一折之后测试集上的结果
        metrics = train_model()
        list_result.append(metrics)
        print('\n完成第{}折的实验结果：{}\n'.format(j, metrics))

        # 删除当前存储数据文件
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
            print('删除文件夹：', args.data_dir)
        # 将存储文件基本目录更新到原来样子
        args.output_dir = out_current_path
        args.data_dir = data_current_path
    print('完成所有划分，数据{}结果如下：\n'.format(args.data_type))
    result_v = ''
    for index, result in enumerate(list_result):
        strr = ''
        for key, value in result.items():
            strr += key + '\t'
            result_v += str(value) + '\t'
        result_v += '\n'
    print(strr)
    print(result_v)
    results_path = os.path.join(args.results_dir,
                                '{}_{}_{}_{}_results.csv'.format(args.model_encdec, args.data_type, args.att_type,
                                                                 args.Gpu_num))
    average_result = dict()
    for key, value in list_result[0].items():
        average_result[key] = 0.0
    for index, result in enumerate(list_result):
        for key, value in result.items():
            average_result[key] += value/10 #因为是十折交叉，所以除以10
    for key, value in average_result.items():
        average_result[key] = np.around(average_result[key],decimals=4)
    strr_ave = '\n'
    result_ave = ''
    for aver_key, aver_value in average_result.items():
        result_ave += aver_key + '_ave\t'
    result_ave += '\n'
    for aver_key, aver_value in average_result.items():
        result_ave += str(aver_value) + '\t'
    strr_ave = strr_ave + '\n' + result_ave
    print(strr_ave)

    f = open(results_path, 'w')
    f.write(strr)
    f.write('\n')
    f.write(result_v)
    f.write(strr_ave)
    f.close()
    result_dict = dict()
    for key, value in list_result[0].items():
        result_dict[key] = []
    for index, result in enumerate(list_result):
        for key, value in result.items():
            result_dict[key].append(value)
    for key, aver_key in zip(list_result[0].keys(), average_result.keys()):
        result_dict[key].append(aver_key+'_ave')
    for key, value in zip(list_result[0].keys(), average_result.values()):
        result_dict[key].append(value)
    df = pd.DataFrame(result_dict)
    new_results_path = os.path.join(args.results_dir,
                                '{}_{}_{}_results.csv'.format(args.model_encdec, args.data_type,
                                                                 args.Gpu_num))
    df.to_csv(new_results_path,index=False)
if __name__ == "__main__":
    # train_model(gpu_num = 0, split_num= 0)
    if args.ten_fold:
        main2()
    else:
        main()
