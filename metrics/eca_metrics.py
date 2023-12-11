from collections import OrderedDict, defaultdict, Counter
import numpy as np
import time
import pickle
import os

tags = ['O', 'B-C', 'B-E', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise','B-CS','I-CS', 'B-ES','I-ES']
# index: 0,    1,     2,    3,    4,     5,    6      7          8        9         10          11          12
label2id = OrderedDict()
# id2label = OrderedDict()
for idx, item in enumerate(tags):
    label2id[item] = idx
    # id2label[idx] = item
label2id['C'] = 1
label2id['I-C'] = 1
label2id['E'] = 2
label2id['I-E'] = 2

def get_sub_emo_metrics(pre_label_l,tru_label_l):
    num_correct_spans = 0
    num_true_spans = 0
    num_pred_spans = 0
    for pre_label, tru_label in zip(pre_label_l, tru_label_l):
        pre_spans = get_sub_emo_span(pre_label)
        tru_spans = get_sub_emo_span(tru_label)
        num_pred_spans += len(pre_spans)
        num_true_spans += len(tru_spans)
        for tru_span in tru_spans:
            for pre_span in pre_spans:
                if tru_span == pre_span:
                    num_correct_spans += 1
    p = num_correct_spans / num_pred_spans if num_pred_spans > 0 else 0
    r = num_correct_spans / num_true_spans if num_true_spans > 0 else 0
    f = (2 * p * r) / (p +r) if (p+r)>0 else 0
    return (p,r,f)
def get_real_sub_emo_metrics(pre_label_l,tru_label_l):
    num_correct_spans = 0
    num_true_spans = 0
    num_pred_spans = 0
    for pre_label, tru_label in zip(pre_label_l, tru_label_l):
        pre_spans = get_real_sub_emo_span(pre_label)
        tru_spans = get_real_sub_emo_span(tru_label)
        num_pred_spans += len(pre_spans)
        num_true_spans += len(tru_spans)
        for tru_span in tru_spans:
            for pre_span in pre_spans:
                if tru_span == pre_span:
                    num_correct_spans += 1
    p = num_correct_spans / num_pred_spans if num_pred_spans > 0 else 0
    r = num_correct_spans / num_true_spans if num_true_spans > 0 else 0
    f = (2 * p * r) / (p +r) if (p+r)>0 else 0
    return (p,r,f)
def get_sub_cau_metrics(pre_label_l, tru_label_l):
    num_correct_spans = 0
    num_true_spans = 0
    num_pred_spans = 0
    for pre_label, tru_label in zip(pre_label_l, tru_label_l):
        pre_spans = get_sub_cau_span(pre_label)
        tru_spans = get_sub_cau_span(tru_label)
        num_pred_spans += len(pre_spans)
        num_true_spans += len(tru_spans)
        for tru_span in tru_spans:
            for pre_span in pre_spans:
                if tru_span == pre_span:
                    num_correct_spans += 1
    p = num_correct_spans / num_pred_spans if num_pred_spans > 0 else 0
    r = num_correct_spans / num_true_spans if num_true_spans > 0 else 0
    f = (2 * p * r) / (p +r) if (p+r)>0 else 0
    return (p,r,f)

def get_real_sub_cau_metrics(pre_label_l, tru_label_l):
    num_correct_spans = 0
    num_true_spans = 0
    num_pred_spans = 0
    for pre_label, tru_label in zip(pre_label_l, tru_label_l):
        pre_spans = get_real_sub_cau_span(pre_label)
        tru_spans = get_real_sub_cau_span(tru_label)
        num_pred_spans += len(pre_spans)
        num_true_spans += len(tru_spans)
        for tru_span in tru_spans:
            for pre_span in pre_spans:
                if tru_span == pre_span:
                    num_correct_spans += 1
    p = num_correct_spans / num_pred_spans if num_pred_spans > 0 else 0
    r = num_correct_spans / num_true_spans if num_true_spans > 0 else 0
    f = (2 * p * r) / (p +r) if (p+r)>0 else 0
    return (p,r,f)

def get_sub_cau_span(labels):
    spans = []
    start, end = 0,0
    while start < len(labels):
        if labels[start] == 1:
            end = start
            while end < len(labels) and labels[end]==1:
                end += 1
            spans.append(([start,end-1]))
            start = end
        start = start + 1
    return spans


def get_real_sub_cau_span(labels):
    spans = []
    # start, end = 0,0
    for i in range(len(labels)):
        if labels[i]==1:
            start = i
            end = i
            for j in range(i+1,len(labels)):
                if labels[j] == 2:
                    end += 1
                else:
                    spans.append([start,end])
                    break
    return spans

def get_sub_emo_span(labels):
    spans = []
    start, end = 0,0
    while start < len(labels):
        if labels[start] == 2:
            end = start
            while end < len(labels) and labels[end]==2:
                end += 1
            spans.append(([start,end-1]))
            start = end
        start = start + 1
    return spans


def get_real_sub_emo_span(labels):
    spans = []
    # start, end = 0,0
    for i in range(len(labels)):
        if labels[i]==1:
            start = i
            end = i
            for j in range(i+1,len(labels)):
                if labels[j] == 2:
                    end += 1
                else:
                    spans.append([start,end])
                    break
    return spans

def get_correct_words(span1, span2): #实验中的片段起始区间是两端闭区间。（1.5）表示1，2，3，4，5 这五个字符
    if span1[1] < span2[0] or span2[1] < span1[0]:
        return 0
    elif span1[0] <= span2[0] and span1[1] >= span2[1]:
        return span2[1] - span2[0] + 1
    elif span2[0]<=span1[0] and span2[1]>=span1[1]:
        return span1[1] - span1[0] + 1
    else:
        if span1[0]<=span2[0]:
            return span1[1] - span2[0] +1
        else:
            return span2[1] - span1[0] + 1

def span2clause(text, span):
    clause_start = span[0] + 1      # 加一是因为模型预测的标签不含[CLS],而这里的文本text里面有[CLS]
    clause_end = span[1] + 1
    while clause_start >= 0:
        clause_start -= 1
        if text[clause_start] == '[CLS]' or text[clause_start] == '[SEP]':
            break
    while clause_end <= len(text) - 1:
        clause_end += 1
        if text[clause_end] == '[SEP]':
            break

    return clause_start+1, clause_end -1 #加一减一是为了保证返回的子句中不包含[CLS]和[SEP],和第一行的加一不冲突

def get_cause_span(tag, real_tags_len):
    """
    tags: numpy array
    real_tags_len: length of a real example plus 2, i.e. [CLS] and [SEP]
    """
    pad_len = real_tags_len - tag.shape[0]
    tag = np.pad(tag, ((0, pad_len), (0, pad_len)), "constant", constant_values=0)  # 长度不足即补零
    cause_spans = []
    start, end = 0,0
    while start < tag.shape[0]:
        if tag[start][start]==1:
            end = start
            while end < tag.shape[0] and tag[end][end]==1:
                end = end + 1
            cause_spans.append([start, end-1])
            start = end
        start = start + 1

    return cause_spans

def get_emotion_span(tag, real_tags_len):
    """
    tags: numpy array
    real_tags_len: length of a real example plus 2, i.e. [CLS] and [SEP]
    """
    pad_len = real_tags_len - tag.shape[0]
    tag = np.pad(tag, ((0, pad_len), (0, pad_len)), "constant", constant_values=0)  # 长度不足即补零
    emotion_spans = []
    start, end = 0, 0
    while start < tag.shape[0]:
        if tag[start][start] == 2:
            end = start
            while end < tag.shape[0] and tag[end][end] == 2:
                end = end + 1
            emotion_spans.append([start, end - 1])
            start = end
        start = start + 1

    return emotion_spans


def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()
def get_accuracy_3(pre_label_l, tru_label_l, examples, pre_label_emotion,pre_labels_emotion_sub, tru_label_emotion,tru_label_emotion_sub, pre_label_cause,pre_labels_cause_sub, tru_label_cause,tru_label_cause_sub): #examples里面的标签不包含[CLS]和最后的[SEP]

    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    pre_label_path = './save_pre_tru/pre_label' + formatted_time + '.pkl'
    tru_label_path = './save_pre_tru/tru_label' + formatted_time + '.pkl'
    examples_path = './save_pre_tru/examples' + formatted_time + 'pkl'
    if not os.path.exists("./save_pre_tru"):
        os.mkdir("./save_pre_tru")
    examples_list = []
    for index, item in enumerate(examples):
        example_dict = {}
        example_dict['text_a'] = item.text_a
        example_dict['DocId'] = item.docid
        example_dict['labels'] = item.labels
        examples_list.append(example_dict)

    saveList(pre_label_l, pre_label_path)
    saveList(tru_label_l, tru_label_path)
    saveList(examples_list, examples_path)
    assert len(pre_label_l) == len(tru_label_l) == len(examples)
    tru_example_label = [] # 存储从example获取的标签
    example_text = []
    for index, item in enumerate(examples):
        single_example_label = np.array([label2id[lb] for row in item.labels for lb in row]).reshape(len(item.labels), len(item.labels))
        tru_example_label.append(single_example_label)
        text_a = item.text_a
        text_a.append('[SEP]')
        text_a.insert(0,'[CLS]')
        example_text.append(text_a)

    correct_triplets = 0
    correct_clauses = 0
    all_triplets = 0

    tru = 0
    pre = 0
    tru_clauses =  0
    pred_clauses = 0

    words_p = 0.0
    words_r = 0.0

    # subtask metrics
    correct_emo_spans = 0
    pred_emo_spans = 0
    tru_emo_spans = 0

    correct_cau_spans = 0
    pred_cau_spans = 0
    tru_cau_spans = 0

    correct_pairs = 0
    pred_pairs = 0
    tru_pairs = 0

    for pre_label, tru_label, example_label, text in zip(pre_label_l, tru_label_l, tru_example_label, example_text):
        assert len(pre_label) == len(tru_label)
        pred_cause, tru_cause = get_cause_span(pre_label, len(example_label)+2), get_cause_span(tru_label, len(example_label)+2)
        pred_emotion, tru_emotion = get_emotion_span(pre_label, len(example_label)+2), get_emotion_span(tru_label, len(example_label)+2)

        pred_triplets = []
        tru_triplets = []
        pred_clause_triplets = []
        tru_clause_triplets = [] #存放所有的真实子句三元组
        polarity_list = [label2id[emo] for emo in ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']]
        # getting true triplets
        for tru_emo in tru_emotion:
            tru_emo_start = tru_emo[0]
            tru_emo_end = tru_emo[1]
            polarity = None  # 表示三元组的极性
            polarity_arrys = np.empty(0)  # 存放所有可能预测的极性
            cause_spans = []    # 存放所有与该情绪配对的原因片段

            cause_clauses = [] #存放所有与该情绪配对的原因子句

            for tru_cau in tru_cause:
                tru_cau_start = tru_cau[0]
                tru_cau_end = tru_cau[1]
                # 判断大小的目的是只用矩阵的上三角
                if tru_cau_start < tru_emo_start:
                    polarity_arry = tru_label[tru_cau_start:tru_cau_end+1, tru_emo_start:tru_emo_end+1]
                else:
                    polarity_arry = tru_label[tru_emo_start:tru_emo_end + 1, tru_cau_start:tru_cau_end + 1]
                if any([polarity_arry[i][j] in polarity_list for i in range(polarity_arry.shape[0]) for j in range(polarity_arry.shape[1])]):

                    # cause_clauses.append(a_cause_clause)
                    #
                    # cause_spans.append((tru_cau_start, tru_cau_end))
                    # polarity_arrys = np.concatenate((polarity_arrys, polarity_arry.reshape(-1)))
                    polar_count = Counter(polarity_arry.reshape(-1))
                    sorted_polar_count = sorted(polar_count.items(), key=lambda x:x[1], reverse=True)
                    for polar in sorted_polar_count:
                        if polar[0] in polarity_list:  # 确保获得的极性是极性
                            polarity = polar[0]
                            break
                    if polarity is not None:  # polarity 不none，表示
                        a_true_triplet = [(tru_emo_start, tru_emo_end), (tru_cau_start, tru_cau_end), polarity]

                        emo_clause = span2clause(text, (tru_emo_start, tru_emo_end))
                        a_cause_clause = span2clause(text, (tru_cau_start, tru_cau_end))
                        a_true_clause_triplet = [emo_clause, a_cause_clause, polarity]

                        all_triplets += 1
                        tru_triplets.append(a_true_triplet)
                        tru_clause_triplets.append(a_true_clause_triplet)

        for pred_emo in pred_emotion:
            pred_emo_start = pred_emo[0]
            pred_emo_end = pred_emo[1]
            pred_polarity = None
            pred_polarity_arrys = np.empty(0)
            pred_cause_spans = []

            pred_cause_clauses = []

            for pred_cau in pred_cause:
                pred_cau_start = pred_cau[0]
                pred_cau_end = pred_cau[1]
                # 这里判断大小，是为了只用到矩阵的上三角
                if pred_cau_start<pred_emo_start:
                    pred_polarity_arry = pre_label[pred_cau_start:pred_cau_end + 1, pred_emo_start:pred_emo_end + 1]
                else:
                    pred_polarity_arry = pre_label[pred_emo_start:pred_emo_end + 1, pred_cau_start:pred_cau_end + 1]
                if any([pred_polarity_arry[i][j] in polarity_list for i in range(pred_polarity_arry.shape[0]) for j in range(pred_polarity_arry.shape[1])]):

                    pred_polar_count = Counter(pred_polarity_arry.reshape(-1))
                    pred_sorted_polar_count = sorted(pred_polar_count.items(), key=lambda x:x[1],reverse=True)

                    for pred_polar in pred_sorted_polar_count:
                        if pred_polar[0] in polarity_list:
                            pred_polarity = pred_polar[0]
                            break
                    if pred_polarity is not None != 0:
                        pred_triplets.append([(pred_emo_start, pred_emo_end), (pred_cau_start, pred_cau_end), pred_polarity])
                        pred_emotion_clause = span2clause(text, (pred_emo_start, pred_emo_end))
                        a_pred_cause_clause = span2clause(text, (pred_cau_start, pred_cau_end))
                        pred_clause_triplets.append([pred_emotion_clause, a_pred_cause_clause, pred_polarity])

        # 统计预测三元组正确之个数
        for tru_tpl in tru_triplets:
            for pred_tpl in pred_triplets:
                if tru_tpl[0] == pred_tpl[0] and tru_tpl[1] == pred_tpl[1] and tru_tpl[2] == pred_tpl[2]:
                    correct_triplets += 1
                if tru_tpl[0] == pred_tpl[0] and tru_tpl[1] == pred_tpl[1]:
                    correct_pairs += 1
        for tru_tpl in tru_clause_triplets:
            for pred_tpl in pred_clause_triplets:
                if tru_tpl[0] == pred_tpl[0] and tru_tpl[1] == pred_tpl[1]: # 子句级别只统计情绪原因对，不统计极性
                    correct_clauses += 1
        tru += len(tru_triplets)
        pre += len(pred_triplets)
        tru_clauses += len(tru_triplets)
        pred_clauses += len(pred_triplets)
        tru_pairs += len(tru_triplets)
        pred_pairs += len(pred_triplets)

        # word-level emotion-cause-pair metrics
        correct_emo_words = 0
        pred_emo_words = 0
        tru_emo_words = 0
        correct_cause_words = 0
        pred_cause_words = 0
        tru_cause_words = 0
        for tru_tpl in tru_triplets:
            for pred_tpl in pred_triplets:
                if tru_tpl[2] == pred_tpl[2]:
                    correct_emo_words += get_correct_words(tru_tpl[0],pred_tpl[0])
                    correct_cause_words += get_correct_words(tru_tpl[1],pred_tpl[1])

        for pred_tpl in pred_triplets:
            pred_emo_words += pred_tpl[0][1] - pred_tpl[0][0] + 1
            pred_cause_words += pred_tpl[1][1] - pred_tpl[1][0] + 1
            # for pre_cau in pred_tpl[1]:
            #     pred_cause_words += pre_cau[1] - pre_cau[0] + 1

        for tru_tpl in tru_triplets:
            tru_emo_words += tru_tpl[0][1] - tru_tpl[0][0] + 1
            tru_cause_words += tru_tpl[1][1] - tru_tpl[1][0] + 1
        emo_words_p = correct_emo_words / pred_emo_words if pred_emo_words >0 else 0
        emo_words_r = correct_emo_words / tru_emo_words if tru_emo_words >0 else 0
        cause_words_p = correct_cause_words / pred_cause_words if pred_cause_words > 0 else 0
        cause_words_r = correct_cause_words / tru_cause_words if tru_cause_words > 0 else 0

        words_p += (emo_words_p + cause_words_p) / 2.0
        words_r += (emo_words_r + cause_words_r) / 2.0

        # 根据主任务的矩阵中对角线的元素，获取子任务的相关指标，包括有:情绪片段prf，原因片段prf，情绪原因片段对prf

        tru_emo_spans += len(tru_emotion)
        pred_emo_spans += len(pred_emotion)
        tru_cau_spans += len(tru_cause)
        pred_cau_spans += len(pred_cause)

        for tru_emo in tru_emotion:
            for pred_emo in pred_emotion:
                if tru_emo == pred_emo:
                    correct_emo_spans += 1
        for tru_cau in tru_cause:
            for pred_cau in pred_cause:
                if tru_cau == pred_cau:
                    correct_cau_spans += 1

    w_p = words_p / pre if pre > 0 else 0
    w_r = words_r / tru if tru > 0 else 0
    w_f = (2 * w_p * w_r) / (w_p + w_r) if (w_p + w_r) > 0 else 0

    p = correct_triplets / pre if pre > 0 else 0
    r = correct_triplets / tru if tru > 0 else 0
    f = (2 * p * r) / (p + r) if (p + r) > 0 else 0

    clause_p = correct_clauses / pred_clauses if pred_clauses > 0 else 0
    clause_r = correct_clauses / tru_clauses if tru_clauses > 0 else 0
    clause_f = (2 * clause_p * clause_r) / (clause_p + clause_r) if (clause_p + clause_r) > 0 else 0

    result = {'tpl_p': np.around(p, decimals=4), 'tpl_r': np.around(r, decimals=4),
                  'tpl_f': np.around(f, decimals=4),
                  'clause_p': np.around(clause_p, decimals=4), 'clause_r': np.around(clause_r, decimals=4),
                  'clause_f': np.around(clause_f, decimals=4),
                  }
        # print('\n')
        # print(result)
    return result