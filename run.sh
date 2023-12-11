#! /bin/bash
# CUDA_VISIBLE_DEVICES=1 nohup python -u run_bert_based_model.py --Gpu_num 1 --data_type merge_ch --model_name_or_path bert_base_chinese --model_encdec bert2softmax --train_max_seq_length 150 --eval_max_seq_length 150 --train_batch_size 8 --eval_batch_size 8 --num_train_epochs 20 --crf_learning_rate 1e-2 --ten_fold True --do_eval False >> result_new_metric.log 2>&1 &
nohup python -u run_bert_based_model.py --Gpu_num 0 --data_type merge_ch --model_name_or_path bert_base_chinese --model_encdec bert2softmax --train_max_seq_length 150 --eval_max_seq_length 150 --train_batch_size 2 --eval_batch_size 1 --num_train_epochs 20 --crf_learning_rate 1e-2 --ten_fold True --do_eval False >> result_15kl.log 2>&1 &
