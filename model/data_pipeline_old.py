'''
input:
    train.txt: ([i1, i2, ...],[t1, t2, ...]) len = 13
    test.txt

add neg 每一个epoch都随机挑选

generate batch 
    users
    candidates

   users: namedtuple
       users.history: (batch_size, n_node)
       users.timestp
       users.adj    : (batch_size, n_node, n_node)
   candidates: namedtuple
       candidates.pos: (batch_size, 1)
       candidates.neg:(batch_size, 1)
'''

import numpy as np
import torch
from collections import namedtuple
import config

batch_size = config.train['batch_size']
hst_size = config.train['history_size']
from collections import defaultdict 
one_time_check = defaultdict(lambda:1)

def neg_generator(one_sample:list, id_max, n_neg):

    neg_id_list = []
    while len(neg_id_list) < n_neg:
        neg_id = np.random.randint(1, id_max+1)
        if neg_id in one_sample:
            continue
        else:
            neg_id_list.append(neg_id)
    return neg_id_list


def generate_batch_train(path, batch_size, device, n_neg=5):
    '''
    path: '/home/.../data/steam/'

    output: 
        users: namedtuple
           users.history: (batch_size, n_node)
           users.timestp: (batch_size, n_node)
           users.adj    : (batch_size, n_node, n_node)
        candidates: namedtuple
           candidates.pos: (batch_size, 1)
           candidates.neg: (batch_size, n_neg)
           candidates.stp: (batch_size, 1)
    '''
    ratio = None

    n_item = config.dataset[config.dataset_choice]['num_items_all']

    if ratio in [0, 0.05, 0.1, 0.15, 0.2]:
        f      = open(path+'train_adversial_'+str(ratio)+'.txt', 'r')
    else:
        f      = open(path+'train.txt', 'r')
    f_attr = open(path+'attr_train.txt', 'r')
    while True:

        Users = namedtuple('User', ['history', 'timestp', 'adj'])
        Candidates = namedtuple('Candidates', ['pos', 'neg', 'stp'])

        '''
        history = torch.zeros((batch_size, hst_size), dtype=torch.long).to(device)
        timestp = torch.zeros((batch_size, hst_size), dtype=torch.long).to(device)
        _adj    = torch.ones((hst_size, hst_size)) - torch.eye(hst_size)
        adj     = _adj.repeat(batch_size, 1).view((batch_size, hst_size, hst_size)).to(device)
        '''
        history = []
        timestp = []
        adj     = []

        pos     = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
        neg     = torch.zeros((batch_size, n_neg), dtype=torch.long).to(device)
        cdd_stp = torch.zeros((batch_size, 1), dtype=torch.long).to(device)

        len_batch = 0
        while(len_batch < batch_size):
            
            try:
                sample, _stp  = eval(f.readline())
                attr_itm_list = eval(f_attr.readline())
                assert len(sample) == hst_size + 1
            except:
                f      = open(path+'train.txt', 'r')
                f_attr = open(path+'attr_train.txt', 'r')
                sample, _stp = eval(f.readline())
                attr_itm_list = eval(f_attr.readline())

            stp = convert_stp(_stp)
            #add stp of attr
            stp_attr = []
            for i in range(len(attr_itm_list)):
                _stp_itm = torch.tensor([stp[w] for w in attr_itm_list[i][1]], dtype=torch.float)
                stp_attr.append(torch.mean(_stp_itm))
            stp_attr = torch.tensor(stp_attr)

            '''
            if one_time_check['stp']: 
                print(f'(one time check) stp:\n{stp}')
                one_time_check['stp'] = 0
            '''

            attr_train = [w[0] + n_item for w in attr_itm_list]
            hst_train = torch.tensor(sample[:hst_size] + attr_train)

            stp_train = stp[:hst_size]
            stp_train = torch.cat((stp_train, stp_attr))

            history.append(hst_train)
            timestp.append(stp_train)
            adj.append(generate_adj(sample[:hst_size], attr_itm_list))

            pos[len_batch] = torch.tensor([sample[-1]])
            neg[len_batch] = torch.tensor([neg_generator(sample, n_item, n_neg)])
            cdd_stp[len_batch] = stp[-1:]

            len_batch += 1

        history, timestp, adj = pad_for_input(history, timestp, adj)
        history = history.to(device)
        timestp = timestp.to(device)
        adj     = adj.to(device)

        users = Users._make([history, timestp, adj])
        candidates = Candidates._make([pos, neg, cdd_stp])

        yield users, candidates

def generate_batch_test(path, device):
    '''
    output: 
        users: namedtuple
           users.history: (batch_size, n_node)
           users.timestp
           users.adj    : (batch_size, n_node, n_node)
        candidates: namedtuple
           candidates.cdd: (batch_size, 1)
           candidates.stp:(batch_size, 1)
    '''
    test_file = path + '/test.txt'
    attr_file = path + '/attr_test.txt'
    test_cdd_file = path + '/test_candidate_1_50.npy'

    n_item = config.dataset[config.dataset_choice]['num_items_all']

    test_all = open(test_file, 'r')
    attr_all = open(attr_file, 'r')
    test_cdd_all = np.load(test_cdd_file)

    for test, attr, test_cdd in zip(test_all, attr_all, test_cdd_all):

        sample, _stp = eval(test)
        attr_itm_list = eval(attr)
        assert sample[-1]==test_cdd[0]
        
        batch_size = len(test_cdd)

        stp = convert_stp(_stp) # float
        #add stp of attr
        stp_attr = []
        for i in range(len(attr_itm_list)):
            _stp_itm = torch.tensor([stp[w] for w in attr_itm_list[i][1]], dtype=torch.float)
            stp_attr.append(torch.mean(_stp_itm))
        stp_attr = torch.tensor(stp_attr).long()

        '''
        if one_time_check['stp']: 
                print(f'(one time check) stp:\n{stp}')
                one_time_check['stp'] = 0
        '''

        attr_test = [w[0] + n_item for w in attr_itm_list]
        hst_test = torch.tensor(sample[:hst_size] + attr_test).long()
        stp_test = stp[:hst_size].long()
        stp_test = torch.cat((stp_test, stp_attr))

        history = hst_test.repeat(batch_size, 1).to(device)
        timestp = stp_test.repeat(batch_size, 1).to(device)
        _adj    = generate_adj(sample[:hst_size], attr_itm_list)
        adj     = _adj.repeat(batch_size, 1, 1).to(device)

        cdd     = torch.tensor(test_cdd, dtype=torch.long).view((batch_size, 1)).to(device)
        cdd_stp = stp[-1:].long().repeat(batch_size, 1).to(device)
        label   = [1]*1 + [0]*50

        Users = namedtuple('User', ['history', 'timestp', 'adj'])
        Candidates = namedtuple('Candidates', ['cdd', 'stp', 'label'])

        users = Users._make([history, timestp, adj])
        candidates = Candidates._make([cdd, cdd_stp, label])

        yield users, candidates

def convert_stp(stp):
    '''
    stp: []
    return torch.tensor
    '''

    stp = torch.tensor(stp).float()
    #stp = (stp - stp[0])/(3600) + 1 # position for pad item is 0
    if stp[-1]-stp[0] > 0:
        stp = stp - stp[0]
        if torch.max(stp) > 50: stp = stp/torch.max(stp)*50 + 1
        else: stp += 1
    else:
        stp = torch.zeros_like(stp) + 1
    return stp

    '''
    def normalize(array):
        if array[-1]-array[0] > 0:
            return (array-np.mean(array))/(array[-1]-array[0])
        else:
            return np.zeros_like(array)
    def sigmoid(array):
        return 1/(1 + np.exp(-array))
    return sigmoid(normalize(stp))
    '''

def generate_adj(hst, attr_item_list):
    '''
    input:
        hst: [itm_id]
        stp: [stp]
        attr_item_list:[(attr_id,[itm_id_idx,...]),]
    output:
        stp_new
        adj
    '''
    
    dim_hst = len(hst)
    dim_attr = len(attr_item_list)
    dim = dim_hst + dim_attr
    adj = torch.zeros((dim, dim))

    adj[:dim_hst, :dim_hst] = torch.ones((dim_hst,dim_hst)) - torch.eye(dim_hst)

    for i in range(len(attr_item_list)):
        for idx in attr_item_list[i][1]:
            adj[i+dim_hst, idx] = 1
            adj[idx, i+dim_hst] = 1

    return adj

def pad_for_input(_history, _timestp, _adj):
    '''
    input:
        history: [torch.tensor([]),...] [#all]
        timestp: [torch.tensor([]),...] [#all]
        adj    : [torch.tensor([[]]),...]
    output:
        history: (batch_size, train_size_withpad), dtype=torch.long
        timestp: (batch_size, train_size_withpad), dtype=torch.long
        adj:     (batch_size, train_size_withpad, train_size_withpad), dtype=torch.long
    '''

    length  = max([len(w) for w in _history])
    n_batch = len(_history)
    history = torch.zeros((n_batch, length), dtype=torch.long)
    timestp = torch.zeros((n_batch, length), dtype=torch.long)
    adj     = torch.zeros((n_batch, length, length), dtype=torch.long)

    for i in range(n_batch):

        l = len(_history[i])
        history[i][:l] = _history[i] + 1 # start from 1 ({items, attrs})
        timestp[i][:l] = _timestp[i]
        adj[i][:l,:l]  = _adj[i]

    return history, timestp, adj


if __name__=='__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)

    #u, c = next(generate_batch_train('./data/movielens/', batch_size = 10, device = device))

    #print(u.history)
    #print()
    #print(u.timestp)
    #print(c.stp)
    #print(c.stp.dtype)
    #print()
    #print(u.adj)

    u, c = next(generate_batch_test('./data/movielens/',device = device))

    print(u.history)
    print()
    print(u.timestp)
    print()
    print(c.cdd)
    print()
    print(c.stp)
    print()
    print(c.label)
    print()







