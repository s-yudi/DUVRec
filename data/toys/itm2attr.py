import gzip
import pickle

path = './meta_Toys_and_Games.json.gz'
data_dir = './'

with open(data_dir + 'id2itm.pkl', 'rb') as f:
    id2itm = pickle.load(f)

itm_set = set(id2itm.values())
itm2attr = {}

for w in gzip.open(path, 'r'):

    w = eval(w)
    if "asin" not in w or w["asin"] not in itm_set: continue
    itm2attr[w["asin"]] = w

with open(data_dir + 'itm2attr.pkl', 'wb') as f:
    pickle.dump(itm2attr, f, pickle.HIGHEST_PROTOCOL)