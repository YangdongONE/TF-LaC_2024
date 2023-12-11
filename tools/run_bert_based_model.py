#encoding=utf-8
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
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger
from tools.func import loadList, saveList

from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from models.bert_for_eca import Bert2Crf, Bert2Gru
from models.albert_for_ner import AlbertCrfForNer

from processors_eca.utils_eca import EcaTokenizer
from processors_eca.eca_seq import convert_examples_to_features
from processors_eca.eca_seq import eca_processors as processors, batch_generator # ner_processors = {"cner": CnerProcessor,'cluener':CluenerProcessor, 'eca':ECAProcessor}
# from processors_eca.eca_seq import collate_fn

from metrics.eca_metrics import get_prf #需要重新写，还没有写
from tools.finetuning_argparse_eca import get_argparse #需要修改一个参数，或者重新写一个类
# from processors_eca.split_seq_data import split
import numpy as np
import os

args = get_argparse().parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Gpu_num)


if args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2crfAtt' or args.model_encdec == 'bert2crfCoAtt':
    MODEL_CLASSES = {
        ## bert ernie bert_wwm bert_wwwm_ext
        'bert': (BertConfig, Bert2Crf, EcaTokenizer)
    }
elif args.model_encdec == 'bert2gru' or args.model_encdec == 'bert2gruAtt' or args.model_encdec == 'bert2gruCoAtt':
    MODEL_CLASSES = {
        ## bert ernie bert_wwm bert_wwwm_ext
        'bert': (BertConfig, Bert2Gru, EcaTokenizer)
    }

if args.data_type == 'ch':  
    args.model_name_or_path = os.path.join(args.root_path,'bert_base_ch')
elif args.data_type == 'en' or args.data_type == 'sti':
    args.model_name_or_path = os.path.join(args.root_path,'bert_base_en')


def train(args, train_features, model, tokenizer, use_crf):
    """ Train the model """
    # args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    #这里用的是随机数据，下面evalue 的时候用的是顺序数据集
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

    if args.model_encdec == 'bert2crf':
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]

    elif args.model_encdec == 'bert2gru':
        gru_param_optimizer = list(model.decoder.named_parameters())
        # linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in gru_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in gru_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]
    
    if args.model_encdec == 'bert2crfAtt':
        crf_param_optimizer = list(model.crf.named_parameters())
        att_param_optimizer = list(model.Attlayer.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate},
                
                {'params': [p for n, p in att_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in att_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]

    elif args.model_encdec == 'bert2gruAtt':
        gru_param_optimizer = list(model.decoder.named_parameters())
        att_param_optimizer = list(model.Attlayer.named_parameters())
        # linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.learning_rate},

                {'params': [p for n, p in gru_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in gru_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate},
                
                 {'params': [p for n, p in att_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
                {'params': [p for n, p in att_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                'lr': args.crf_learning_rate}
            ]

    t_total = len(train_features)//args.train_batch_size * args.num_train_epochs
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
    tr_loss, logging_loss  = 0.0, 0.0
    pre_result = {}
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    total_step = 0
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_features)//args.train_batch_size, desc='Training')
        #修改lixiangju
        step= 0
        for batch in batch_generator(features = train_features, batch_size=args.train_batch_size,  use_crf = use_crf):
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens , batch_label_ids, batch_raw_labels, batch_example = batch
            model.train()

            if args.model_encdec == 'bert2crfAtt' or args.model_encdec == 'bert2gruAtt':
                batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
                inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2],
                "labels": batch_inputs[4], "testing": False}

            elif args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2gru':
                batch_inputs = tuple(t.to(args.device) for t in batch[0:5])

                inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "labels": batch_inputs[4], "testing": False}

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
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args = args, model = model, tokenizer = tokenizer, use_crf = use_crf)
                       
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

        # train_history.append(loss)
        np.random.seed()
        np.random.shuffle(train_features)
        logger.info("\n")
        # if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", use_crf = False):
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
    
    pre_labels, tru_labels, eval_examples  =[],  [], [] #统计整个eval数据的 标签
    # pbar = ProgressBar(n_total=len(eval_features)//args.train_batch_size, desc='Training')
    step = 0
    for batch in batch_generator(features = eval_features, batch_size=args.train_batch_size,  use_crf = use_crf):
        
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens, batch_label_ids, batch_raw_labels, batch_example = batch
        model.eval()
        if args.model_encdec == 'bert2crfAtt' or args.model_encdec == 'bert2gruAtt':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:9])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "input_e_ids": batch_inputs[4], 
             "attention_e_mask":batch_inputs[5], "token_type_e_ids": batch_inputs[6], "labels": None, "testing": True}

        elif args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2gru':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "labels": None, "testing": True}

        outputs = model(**inputs)
        eval_examples.extend(batch_example)
        out_label_ids = batch[9].tolist() #真实标签
        if use_crf:
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            # eval_loss += tmp_eval_loss.item()
            # nb_eval_steps += 1
            # input_lens = batch_inputs[3].cpu().numpy().tolist() 
            tags = tags.squeeze(0).cpu().numpy().tolist()
            for len_doc, cu_tags, cu_trus in zip(batch_lens, tags, out_label_ids):
                pre_labels.append(cu_tags[0: len_doc])
                tru_labels.append(cu_trus[0: len_doc])
        else:
            tags = outputs[0].detach().cpu().numpy().argmax(-1)
            tags = tags.tolist()
            # out_label_ids = batch[5].tolist() #真实标签
            # pre_labels =[list(p[i]) for i in range(p.shape[0])]
            for len_doc, cu_tags, cu_trus in zip(batch_lens, tags, out_label_ids):
                pre_labels.append(cu_tags[0: len_doc])
                tru_labels.append(cu_trus[0: len_doc])

        step += 1
        pbar(step)
        
    logger.info("\n")
    # eval_loss = eval_loss / nb_eval_steps
    # eval_info, entity_info = metric.result()
    results = get_prf(pre_labels, tru_labels, eval_examples)
    # results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    print('results = ', results)
    return results


def predict(args, model, tokenizer, prefix="", use_crf = False):
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

    pre_labels, tru_labels, test_examples  =[],  [], [] #统计整个eval数据的 标签
    step = 0
    for batch in batch_generator(features = test_features, batch_size=args.train_batch_size,  use_crf = use_crf):
        # batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens, batch_label_ids, batch_raw_labels, batch_example = batch
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_lens, batch_label_ids, batch_raw_labels, batch_example = batch
        model.eval()
        if args.model_encdec == 'bert2crfAtt' or args.model_encdec == 'bert2gruAtt':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:9])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "input_e_ids": batch_inputs[4], 
             "attention_e_mask":batch_inputs[5], "token_type_e_ids": batch_inputs[6], "labels": None, "testing": True}

        elif args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2gru':
            batch_inputs = tuple(t.to(args.device) for t in batch[0:5])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], "labels": None, "testing": True}

        outputs = model(**inputs)
        test_examples.extend(batch_example)
        out_label_ids = batch[9].tolist() #真实标签

        if use_crf:
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            # eval_loss += tmp_eval_loss.item()
            # nb_eval_steps += 1
            # input_lens = batch_inputs[3].cpu().numpy().tolist() 
            tags = tags.squeeze(0).cpu().numpy().tolist()

            for len_doc, cu_tags, cu_trus in zip(batch_lens, tags, out_label_ids):
                pre_labels.append(cu_tags[0: len_doc])
                tru_labels.append(cu_trus[0: len_doc])
        
        else:
            tags = outputs[0].detach().cpu().numpy().argmax(-1)
            tags = tags.tolist()
            # out_label_ids = batch[5].tolist() #真实标签
            # pre_labels =[list(p[i]) for i in range(p.shape[0])]
            for len_doc, cu_tags, cu_trus in zip(batch_lens, tags, out_label_ids):
                pre_labels.append(cu_tags[0: len_doc])
                tru_labels.append(cu_trus[0: len_doc])

        step += 1
        pbar(step)
        
    logger.info("\n")
    # eval_loss = eval_loss / nb_eval_steps
    # eval_info, entity_info = metric.result()
    results = get_prf(pre_labels, tru_labels, test_examples)
    # results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    print('results = ', results)
    
    output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
    test_submit = []
    for pre, tru, exam in zip(pre_labels, tru_labels, test_examples):
        json_d = {}
        json_d['docid'] = exam.docid
        json_d['words'] = exam.text_a
        # json_d['tru_cause'] = get_content(exam.text_a, tru)
        # json_d['pre_cause'] = get_content(exam.text_a, pre)
        json_d['tag_seq'] = " ".join([args.id2label[x] for x in pre])
        test_submit.append(json_d)
    json_to_text(output_submit_file,test_submit)
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
    label_list = processor.get_labels() #[B I O]
    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir) #获取数据，并且 增加了将数据存储为csv文件
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                else args.eval_max_seq_length, #需要测试一下两部分数据的最大长度是什么
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
    #允许重写，也就是每次允许的时候，不用担心里面有文件
    # if os.path.exists(args.output_dir) and os.listdir( #如果存在文件不为空，并且不能覆盖重写，那么就返回错误，在这里程序设置为可以重写
    #         args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #             args.output_dir))

    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1

    if args.model_encdec == 'bert2gru' or args.model_encdec == 'bert2gruAtt' or args.model_encdec == 'bert2gruCoAtt':
        use_crf = False
    elif args.model_encdec == 'bert2crf' or args.model_encdec == 'bert2crfAtt' or args.model_encdec == 'bert2crfCoAtt' :
        use_crf = True

    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), )
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.data_type = args.data_type.lower() #将任务名称设置为小写
    if args.data_type not in processors:  #如果任务名称不在列表中，那么返回错误,列表为 ner_processors = {"cner": CnerProcessor,'cluener':CluenerProcessor, 'eca':ECAProcessor}
        raise ValueError("Task not found: %s" % (args.data_type))
    processor = processors[args.data_type]()
    label_list = processor.get_labels() #这里是获取的标签列表，本任务：【B I O】 现在是【C-B C-I E-B E-I O】
    
    args.id2label = {i: label for i, label in enumerate(label_list)} #获取的是字典{0: O, 1: C-B, 2: C-I, 3: E-B, 4: E-I}
    args.label2id = {label: i for i, label in enumerate(label_list)} #获取的是字典{O:0, C-B:1, C-I: 2, E-B: 3, E-I: 4}
    num_labels = len(label_list) # 标签的个数

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] #BertConfig, BertCrfForNer, EcaTokenizer
    #BertConfig.from_pretrained
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None) #模型
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_features = load_and_cache_examples(args, args.data_type, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_features, model, tokenizer, use_crf = use_crf)
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
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
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
            result = evaluate(args = args, model = model, tokenizer = tokenizer, prefix=prefix, use_crf = use_crf)
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
            metrics = predict(args, model, tokenizer, prefix=prefix, use_crf = use_crf)
    return metrics
        



def main():
    if not os.path.exists(args.results_dir): #如果不存在进行创建
        os.mkdir(args.results_dir)


    list_result = []
    # print(args.split_times)
    out_current_path = copy.deepcopy(args.output_dir) 

    for i in range(args.split_times):
        print("*****************************split_times:{}*******************".format(i))
        args.output_dir = args.output_dir + '{}_{}_{}_{}'.format(args.data_type, args.model_encdec, args.Gpu_num, i) #输出模型文件的位置
        if os.path.exists(args.output_dir): #是否存在输出文件，如果不存在进行创建
            shutil.rmtree(args.output_dir)
            print('删除文件夹：',args.output_dir)
        os.mkdir(args.output_dir)
        #观测数据文件中是否包含数据
        data_current_path = copy.deepcopy(args.data_dir) 
        args.data_dir = args.data_dir  + '{}_{}_{}_{}'.format(args.data_type, args.model_encdec, args.Gpu_num, i)  
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
        os.mkdir(args.data_dir)
        #j加载数据并划分
        # if args.data_type == 'ch':
        data_set = loadList(os.path.join(args.dataset_eca_raw, 'eca_{}.pkl'.format(args.data_type)))

        # if args.data_type == 'en':
        # data_set = loadList(os.path.join(args.dataset_eca_raw, 'eca_en_data.pkl'))
        print('data_set = ', len(data_set))
        #每次划分之前打乱数据集进行数据划分
        np.random.seed()
        np.random.shuffle(data_set)
        train_num = int(len(data_set) * 0.8)
        test_num = int(len(data_set) * 0.1)

        train_data = data_set[0 : train_num]
        test_data = data_set[train_num : train_num + test_num]
        dev_data = data_set[train_num + test_num : ]

        #保存数据到对应的文件夹中
        saveList(train_data, os.path.join(args.data_dir, 'eca_train.pkl'))
        saveList(test_data, os.path.join(args.data_dir, 'eca_test.pkl'))
        saveList(dev_data, os.path.join(args.data_dir, 'eca_dev.pkl'))

        #获取一次划分之后测试集合上的结果
        metrics = train_model()
        list_result.append(metrics)
        print('\n完成当前划分的实验结果：bert_crf_results_{} = {}\n'.format(args.data_type, metrics))

        #删除当前存储数据文件
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
            print('删除文件夹：', args.data_dir)
            
        #将存储文件基本目录更新到原来样子
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
        
    results_path = os.path.join(args.results_dir, '{}_{}_{}_{}_results.csv'.format(args.model_encdec, args.data_type, args.att_type, args.Gpu_num))
    f = open(results_path,'w')
    f.write(strr)
    f.write('\n')
    f.write(result_v)
    f.close()

if __name__ == "__main__":
    # train_model(gpu_num = 0, split_num= 0)
    main()
