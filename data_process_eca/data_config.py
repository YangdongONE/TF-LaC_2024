class data_config:

    def __init__(self):

        """
        1.首先读取所有的数据从xml文件到pkl文件
        2.对数据进行划分，并生成数据字典
        """
        self.raw_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/data_process_eca/ECAEmnlp2016.xml'
        self.pkl_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/data_process_eca/ECAEmnlp_new.pkl'

        #将三组数据进行划分，并转换成数据信息的格式，每一个是一个example，包含必要的属性，并将原来的文本的信息写入csv中
        self.tr_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_tr.pkl'
        self.dev_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_dev.pkl'
        self.te_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_te.pkl'
        #将三部分数据写入到csv中
        self.text_tr_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_text_tr.csv'
        self.text_dev_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_text_dev.csv'
        self.text_te_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/dataset_eca/eca_text_te.csv'


        #获取距离的list以及存储的向量， 和源文件没有关系
        self.wpos_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/data_process_dis/wpos_list.pkl'
        self.cpos_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/data_process_dis/cpos_list.pkl'
        self.pc2v_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/data_process_dis/pc2v.npy'
        self.pw2v_path = '/home/lq/seqtoseqcoling/BERT_CRF/bert_lstm_torch/nertorch/data_process_dis/pw2v.npy'


