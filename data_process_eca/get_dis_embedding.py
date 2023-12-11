import numpy as np
import pickle 

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()
    
wpos_list = list(range(-1000, 1000))
cpos_list = list(range(-100, 100))

wpos_dim = 20
cpos_dim = 20

wpos_vector =np.random.rand(len(wpos_list), wpos_dim).astype(np.float32)
cpos_vector =np.random.rand(len(cpos_list), cpos_dim).astype(np.float32)
wpos_vector[0] = np.zeros((wpos_dim))
cpos_vector[0] = np.zeros((cpos_dim))

saveList(wpos_list, '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATE/data_process_dis/wpos_list.pkl')
saveList(cpos_list, '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATE/data_process_dis/cpos_list.pkl')
save_w_path = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATE/data_process_dis/pw2v.npy'
save_c_path = '/home/lq/seqtoseqcoling/Coling/Seq2Seq4ATE/data_process_dis/pc2v.npy'

np.save(save_w_path, wpos_vector)
np.save(save_c_path, cpos_vector)

print(wpos_vector[0])
print(cpos_vector[0])


