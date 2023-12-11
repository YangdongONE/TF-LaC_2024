import pickle
import numpy as np

save_data_path_tr = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_seq_data_tr.pkl'
save_data_path_te = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_seq_data_te.pkl'
save_data_path_dev = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/input_seq_data_dev.pkl'

word_list_path = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/code/word_list.pkl'
wpos_list_path = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/wpos_list.pkl'
cpos_list_path = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/data_process_dis/cpos_list.pkl'


'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()


def get_emo_dis(para_list, para_str = 'word'):
    """
    para_list
    """

    re_list = [0] * len(para_list) #要返回的列表长度，和原来的列表长度一样
    # for ind, ite in enumerate(para_list):
    start = para_list.index(1) #检索情感词语开始的位置
    end = start + sum(para_list)#结束的位置，但是不包含end的位置
    for i in range(start, end): #情感词语左侧的位置索引
        re_list[i] = 0 #是情感词语的位置，设置为0
    if start != 0: #如果开始的位置不为0，那么就要做差计算位置
        for i in range(0, start): 
            re_list[i] = i - start #从0到情感词语开始的位置，分别求相对距离
    if end != len(para_list):
        for i in range(end, len(para_list)):
            re_list[i] = i - end + 1#这里加1是因为求索引的时候，不包含最后一个索引。
    # print('para_list = ', para_list)
    # print('re_list = ', re_list)

    assert len(para_list) == len(re_list)
    rel = []
    if para_str == 'word':
        rel = [wpos_list.index(item) for item in re_list]
    if para_str == 'clause':
        rel = [cpos_list.index(item) for item in re_list]
    assert len(rel) == len(para_list)
    # print('rel = ', rel)
    return rel, re_list #分别返回一个相对距离的列表，和一个相对距离索引的列表


target = ['O', 'C-B', 'C-I','E-B','E-I']  #这边加了断点，未影响主程序之运行，暂且不管
word_list = loadList(word_list_path)
wpos_list = loadList(wpos_list_path)
cpos_list = loadList(cpos_list_path)

def get_data_example(data):
    example = []
    for i in range(len(data)):
        data_ID = data[i]['docID']
        data_x = data[i]['content_Data']
        data_e = data[i]['emotion_Data']
        
        data_y = [target.index(item) for item in data[i]['target_Data'].split()]
        data_y.insert(0,0)
        data_y.append(0)#因为在起始位置和结束位置分别填充了

        data_len_c = data[i]['data_len'] 
        data_len_c[0] = data_len_c[0] + 1#因为要填充字符【CLS】
        data_len_c[-1] = data_len_c[-1] + 1 #因为在末尾填充字符【SEP】

        #相对于情感词语
        emotion_index = data[i]['emotion_index'] #情感词语的位置为1,其余位置为0
        emotion_index.insert(0,0) #因为下面在token的时候最开始加入了 cls 结束的位置加了sep所以这里引入0
        emotion_index.append(0)
        emotion_dis_ids, emo_dis = get_emo_dis(emotion_index, 'word')#获取距离情感词语的距离的信息
        # print('ooooo = ',emotion_dis)

        #相对于情感子句
        ec_index = data[i]['ec_index'] #情感子句位置处为1，其余位置为0
        ec_dis_ids, ec_dis = get_emo_dis(ec_index, 'clause')
        ec_dis_content = []
        for i in range(len(data_len_c)):
            ec_dis_content.extend([ec_dis_ids[i]] * data_len_c[i])

        # print('ccccc = ', ec_dis) 
        token_ids = [word_list.index(item) for item in list(data_x)]
        token_ids.insert(0,word_list.index('[CLS]'))
        token_ids.append(word_list.index('[SEP]'))

        assert len(token_ids) == len(data_y) == len(emotion_index)
        token_ids_emo = [word_list.index(item) for item in list(data_e)]
            
        output = {
                "token_ids": token_ids,
                "target_id": data_y,
                "token_ids_emo": token_ids_emo,
                "dataID":data_ID,
                "data_len": data_len_c,
                "data_x":data_x,
                "data_e":data_e,
                "emotion_index": emotion_index,
                "ec_index": ec_index,
                "ew_dis_ids": emotion_dis_ids,
                "ec_dis_ids": ec_dis_ids,#长度等于子句的个数
                "ew_dis":emo_dis,
                "ec_dis": ec_dis,
                "ec_dis_content": ec_dis_content #将情感子句的索引扩充到词语上
            }
        example.append(output)
        # print(example)
    return example


def pad_data(examples, path):

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = np.zeros(shape = (len(indice), max_length))
        for index, item in enumerate(indice):
            for indexj, itemj in enumerate(item):
                pad_indice[index][indexj] = itemj
        return pad_indice.astype(np.int32)
    
    token_ids = [data["token_ids"] for data in examples] #context的索引
    max_length = max([len(t) for t in token_ids])
    print('max_length = ', max_length)
    target_ids = [data["target_id"] for data in examples] #label的索引
    
    ew_ids_pos = [data["ew_dis_ids"] for data in examples] #获取词语的索引
    ew_ids_pos_padded = padding(ew_ids_pos, max_length) #扩充情感词语相对位置

    ew_pos = [data["ew_dis"] for data in examples] #获取词语的相对距离
    ew_pos_padded = padding(ew_pos, max_length, pad_idx=int(1e12)) #扩充情感词语相对距离

    ec_dis_content = [data["ec_dis_content"] for data in examples] #获取子句的相对距离的索引，此时长度等于文档中词语的长度
    ec_dis_content_pad = padding(ec_dis_content, max_length, pad_idx=int(1e12))

    #扩展contex
    token_ids_padded = padding(token_ids, max_length) 
    #扩展target
    target_ids_padded = padding(target_ids, max_length)

    #扩展情感数据
    token_ids_emo = [data["token_ids_emo"] for data in examples]
    max_length_emo = max([len(t) for t in token_ids_emo])
    token_ids_emo_padded = padding(token_ids_emo, max_length_emo) 
    
    np.savez(path,x = token_ids_padded, x_e = token_ids_emo_padded, y=target_ids_padded, ew_ids_pos_padded = ew_ids_pos_padded, ew_pos_padded = ew_pos_padded,ec_dis_content_pad = ec_dis_content_pad)
    return token_ids, target_ids, token_ids_emo

data_tr = loadList(save_data_path_tr)
data_te = loadList(save_data_path_te)
data_dev = loadList(save_data_path_dev)

example_tr = get_data_example(data_tr)
example_te = get_data_example(data_te)
example_dev = get_data_example(data_dev)

print(example_tr[0])

path_tr = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/code/data_tr.npz'
path_te = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/code/data_te.npz'
path_dev = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/code/data_dev.npz'

print('start save data')
print('save tr')
saveList(example_tr, '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/code/examples_tr.pkl')
print('save te')
saveList(example_te, '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/code/examples_te.pkl')
print('save dev')
saveList(example_dev, '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATEdis/code/examples_dev.pkl')

print('save tr npy')
pad_data(example_tr, path_tr)
print('save te npy')
pad_data(example_te, path_te)
print('save dev npy')
pad_data(example_dev, path_dev)

print('run  over')