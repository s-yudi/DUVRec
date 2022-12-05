
'''
start from 0
output attr2id.pkl
'''

import pickle
itm2attr = pickle.load(open('itm2attr.pkl', 'rb'))
# 'tags','genres'
attr2id = {}
for itm in itm2attr:
    if 'genres' in itm2attr[itm]:
        attr_list = itm2attr[itm]['genres']   
        for attr in attr_list:
            if attr not in attr2id:
                attr2id[attr] = len(attr2id) 
with open('attr2id.pkl', 'wb') as f:
    pickle.dump(attr2id, f, pickle.HIGHEST_PROTOCOL)


from tqdm import tqdm 
import numpy as np

mode = 'test'
f_output = open(f'attr_{mode}.txt', 'w')
'''
attr_train.txt: [(attr_id,[itm_id_idx,...]), ...]
attr_id starts from 0
'''

f_data = open(f'{mode}.txt', 'r')

id2itm = pickle.load(open('id2itm.pkl', 'rb'))
itm2attr = pickle.load(open('itm2attr.pkl', 'rb'))
attr2id = pickle.load(open('attr2id.pkl', 'rb'))

his_size = 12
for row in tqdm(f_data, total=float('inf')):

    his, stp = eval(row.strip())

    #attr_stp_dic = {}
    attr_itm_dic = {}
    for i in range(his_size):
        itm = his[i]
        _stp = stp[i]
        if 'genres' not in itm2attr[id2itm[itm]]: continue
        attrs = [attr2id[w] for w in itm2attr[id2itm[itm]]['genres']]
        for attr in attrs:
            if attr not in attr_itm_dic:
                attr_itm_dic[attr] = [i]
                continue
            attr_itm_dic[attr].append(i)
            #attr_stp_dic[attr].append(_stp)
    
    attr_itm_list, attr_stp_list = [], []
    for attr in attr_itm_dic.keys():
        attr_itm_list.append((attr, attr_itm_dic[attr]))
        #attr_stp_list.append(int(np.mean(attr_stp_dic[attr])))

    f_output.write(str(attr_itm_list)+'\n')

f_data.close()