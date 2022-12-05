'''
ratings.dat
    UserID::MovieID::Rating::Timestamp
movies.dat
    MovieID::Title::Genres
'''

'''
target file:
    train.txt: 
        ith row: ([itm1, itm2, ...],[stp1, stp2, ...]) id start from 1
    test.txt
    test_candidate_1_50.npy
'''

'''
user2asin:{user:([itm1, ...], [stp1,...]}
然后过滤(看一下历史记录长度为多少比较合适)
最后编码
'''
import pickle
from tqdm import tqdm
import numpy as np
np.random.seed(63)

data = {}

with open('ratings.dat', 'rb') as f:

    for row in tqdm(f.readlines()):

        uid, iid, rating, stp = str(row, encoding='utf8').strip().split('::')

        if uid not in data:
            data[uid] = ([],[])
        data[uid][0].append(iid)
        data[uid][1].append(stp)

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

#---

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

'''
for len_history in range(13, 60, 5):
    
    num = 0
    for usr in data:
        #if len(data[usr][0]) >= len_history:
        num += len(data[usr][0])//len_history
    print(f'len_history {len_history} data_size {num}')
'''

import numpy as np

len_sample = 13

all_sample = {}
'''
all_sample[sample_id] = ([itm],[stp]) 
sample_id starts from 0
'''
sample_id = 0
for u in tqdm(data):
    l = len(data[u][0])
    if l < len_sample:
        continue

    ind = np.argsort(np.array(data[u][1]))
    itm_list = list(np.array(data[u][0])[ind])
    stp_list = list(np.array(data[u][1])[ind])

    for start in range(l//len_sample):

        all_sample[sample_id] = (itm_list[start:start+len_sample] , stp_list[start:start+len_sample])
        sample_id += 1

with open('all_sample.pkl', 'wb') as f:
    pickle.dump(all_sample, f, pickle.HIGHEST_PROTOCOL)

#---

with open('all_sample.pkl', 'rb') as f:
    all_sample = pickle.load(f)

#编号 start from 1
itm2id,id2itm = {},{}
for u in all_sample:
    for itm in all_sample[u][0]:
        if itm not in itm2id:
            id2itm[len(itm2id)+1] = itm
            itm2id[itm] = len(itm2id)+1
with open('id2itm.pkl', 'wb') as f:
    pickle.dump(id2itm, f, pickle.HIGHEST_PROTOCOL)
with open('itm2id.pkl', 'wb') as f:
    pickle.dump(itm2id, f, pickle.HIGHEST_PROTOCOL)

#分割
n_sample = len(all_sample)
idx_all = np.arange(n_sample)
np.random.shuffle(idx_all)

f_train = open('train.txt', 'w')
f_test = open('test.txt', 'w')

str_train, str_test = '', ''

for i in tqdm(idx_all[:n_sample//10]): 
    itm_list = [itm2id[itm] for itm in all_sample[i][0]]
    stp_list = all_sample[i][1]
    str_test += str((itm_list, stp_list)) + '\n'
f_test.write(str_test)
f_test.close()

for i in tqdm(idx_all[n_sample//10:]): 
    itm_list = [itm2id[itm] for itm in all_sample[i][0]]
    stp_list = all_sample[i][1]
    str_train += str((itm_list, stp_list)) + '\n'
f_train.write(str_train)
f_train.close()