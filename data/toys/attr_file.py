'''
start from 0
output attr2id.pkl

data_dir = './'

import pickle

itm2attr = pickle.load(open(data_dir + 'itm2attr.pkl', 'rb'))
id2itm = pickle.load(open(data_dir + 'id2itm.pkl', 'rb'))
# "category"
attr2id = {}
for itm in itm2attr:
    if "category" in itm2attr[itm]:
        attr_list = itm2attr[itm]["category"]
        for attr in attr_list:
            if attr in itm2attr[itm]["feature"] or len(attr)>50:
                continue
            if attr not in attr2id:
                attr2id[attr] = len(attr2id)
with open(data_dir + 'attr2id.pkl', 'wb') as f:
    pickle.dump(attr2id, f, pickle.HIGHEST_PROTOCOL)

print(max(attr2id.values())+1)
print(max(id2itm.keys()))
from tqdm import tqdm

mode = 'train'
f_output = open(data_dir + f'attr_{mode}.txt', 'w')
'''
attr_train.txt: [(attr_id,[itm_id_idx,...]), ...]
attr_id starts from 0
'''

f_data = open(data_dir + f'{mode}.txt', 'r')

id2itm = pickle.load(open(data_dir + 'id2itm.pkl', 'rb'))
itm2attr = pickle.load(open(data_dir + 'itm2attr.pkl', 'rb'))
attr2id = pickle.load(open(data_dir + 'attr2id.pkl', 'rb'))

his_size = 12
for row in tqdm(f_data): #, total=float('inf')

    his, stp = eval(row.strip())

    # attr_stp_dic = {}
    attr_itm_dic = {}
    for i in range(his_size):
        itm = his[i]
        _stp = stp[i]
        if id2itm[itm] not in itm2attr or "category" not in itm2attr[id2itm[itm]]: continue
        attrs = [attr2id[w] for w in itm2attr[id2itm[itm]]["category"] if w in attr2id.keys()]
        for attr in attrs:
            if attr not in attr_itm_dic:
                attr_itm_dic[attr] = [i]
                continue
            attr_itm_dic[attr].append(i)
            # attr_stp_dic[attr].append(_stp)

    attr_itm_list, attr_stp_list = [], []
    for attr in attr_itm_dic.keys():
        attr_itm_list.append((attr, attr_itm_dic[attr]))
        # attr_stp_list.append(int(np.mean(attr_stp_dic[attr])))

    f_output.write(str(attr_itm_list) + '\n')

f_data.close()