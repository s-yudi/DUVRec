'''
source file: dataset.json.gz
Sample review:
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
'''

'''
target file:
    train.txt
    test.txt
    id2itm
'''
import gzip
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
      yield eval(l.decode('UTF-8').replace('true', 'True').replace('false', 'False'))

import numpy as np
from tqdm import tqdm

path = './Toys_and_Games.json.gz'
data_dir = './'

item2id = {}
user2id = {}
data = {}
usernum = 0
itemnum = 0
reviewnum = 0

for review in parse(path):
    reviewnum += 1
    if review["reviewerID"] not in user2id.keys():
        usernum += 1
        user2id[review["reviewerID"]] = usernum
    if review["asin"] not in item2id.keys():
        itemnum += 1
        item2id[review["asin"]] = itemnum
    if user2id[review["reviewerID"]] not in data.keys():
        data[user2id[review["reviewerID"]]] = [[], []]
    data[user2id[review["reviewerID"]]][0].append(item2id[review["asin"]])
    data[user2id[review["reviewerID"]]][1].append(review["unixReviewTime"])
for u in data.keys():
    index = np.argsort(data[u][1])
    data[u][1] = list(np.array(data[u][1])[index])
    data[u][0] = list(np.array(data[u][0])[index])

print(usernum)
print(itemnum)
print(reviewnum)


len_sample = 13
all_sample = {}
sample_id = 0
for u in tqdm(data):
    l = len(data[u][0])
    if l < len_sample:
        continue
    for start in range(l//len_sample):
        all_sample[sample_id] = (data[u][0][start:start+len_sample], data[u][1][start:start+len_sample])
        sample_id += 1

print(len(all_sample))

np.random.seed(63)
import pickle
id2itm = {value:key for key, value in item2id.items()}
with open(data_dir + 'id2itm.pkl', 'wb') as f:
    pickle.dump(id2itm, f, pickle.HIGHEST_PROTOCOL)
n_sample = len(all_sample)
idx_all = np.arange(n_sample)
np.random.shuffle(idx_all)

f_train = open(data_dir + 'train.txt', 'w')
f_test = open(data_dir + 'test.txt', 'w')

str_train, str_test = '', ''

for i in tqdm(idx_all[:n_sample//10]):
    itm_list = all_sample[i][0]
    stp_list = all_sample[i][1]
    str_test += str((itm_list, stp_list)) + '\n'
f_test.write(str_test)
f_test.close()


for i in tqdm(idx_all[n_sample//10:]):
    itm_list = all_sample[i][0]
    stp_list = all_sample[i][1]
    str_train += str((itm_list, stp_list)) + '\n'
f_train.write(str_train)
f_train.close()

