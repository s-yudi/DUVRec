
import gzip
import pickle

with open('itm2id.pkl', 'rb') as f:
    itm2id = pickle.load(f)
with open('itm2id_2.pkl', 'rb') as f:
    itm2id_2 = pickle.load(f)

id2itm = {}
for itm in itm2id:
    try:
        idx = itm2id_2[str(itm2id[itm])]
        id2itm[idx] = itm
    except KeyError:
        continue
with open('id2itm.pkl', 'wb') as f:
    pickle.dump(id2itm, f, pickle.HIGHEST_PROTOCOL)

id2itm = pickle.load(open('id2itm.pkl', 'rb'))
itm_set = set(id2itm.values())
itm2attr = {}
for w in gzip.open('steam_games.json.gz', 'r'):

    w = eval(w)
    if 'id' not in w or w['id'] not in itm_set: continue
    itm2attr[w['id']] = w

with open('itm2attr.pkl', 'wb') as f:
    pickle.dump(itm2attr, f, pickle.HIGHEST_PROTOCOL)

