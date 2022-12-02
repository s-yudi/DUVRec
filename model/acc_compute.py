


import torch 
import numpy as np
import model
import config

dataset    = 'steam' ###
model_name = 'mymodel' ###
epoch_recresult = [w for w in range(50)] ###

#path = '/home/xlx2/mymodel/baseline/SR-GNN-master/pytorch_code/'
#path = f'/home/xlx2/mymodel/recresult/variant/' ### recresult
#path = '/home/xlx2/mymodel/weights/steam_mymodel/'
#path = '/home/xlx2/mymodel/weights/05201416/'# steam_alter0
#path = f'/home/xlx2/mymodel/weights/{dataset}_mymodel/'
#path = f'/home/xlx2/mymodel/weights/05211244/'# movielens_alter0
#path = '/home/xlx2/mymodel/weights/05250323_steam_nassign-5/'# steam_alter0
path = '/home/xlx2/mymodel/weights/05260338_steam_nassign-1/'

if dataset=='steam' : 
    epoch_model = 68
elif dataset=='movielens':
    epoch_model = 144
k = 10

f = open(path + f'acc_{k}.txt', 'w')
_f = ''
'''
hit, mrr 
'''
print(f'{dataset}_{model_name}')
_f += f'{dataset}_{model_name}\n'
for epoch in epoch_recresult:
    test_data = np.load(f'/home/xlx2/mymodel/data/{dataset}/test_candidate_1_50.npy')
    recresult = np.load(path + f'{epoch}_recresult_{model_name}_{dataset}.npy') ###
    hit, mrr = 0, 0

    for rec, truth in zip(recresult, test_data):

        #print(rec)
        #print(truth)
        pos = truth[0]
        #print(np.where(rec==pos))
        idx = np.where(rec==pos)[0][0]+1

        if idx <= k:
            hit += 1.0
            mrr += 1.0 / idx

    n = np.shape(recresult)[0]    
    hit = hit/n
    mrr = mrr/n

    print(f'epoch{epoch} hit@{k}: {hit}, mrr@{k}: {mrr}')
    _f += f'epoch{epoch} hit@{k}: {hit}, mrr@{k}: {mrr}\n'

'''
ild
'''
ILD = []

recmodel = model.MyModel(
                **config.model_args['mymodel']
            )
weight_file = f'/home/xlx2/mymodel/weights/{dataset}_mymodel/mymodel_{dataset}_{epoch_model}.pth.tar'
#weight_file = f'/home/xlx2/mymodel/weights/01171451/mymodel_amazon_movie_65.pth.tar'
# mymodel_steam_72.pth.tar
# 01162129/mymodel_movielens_118.pth.tar
# 01171451/mymodel_amazon_movie_65.pth.tar

recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))

#rec_result = np.load('./diversity_file/1_rec_result_srgnn.npy')
print('*****DIVERSITY*****')
_f += '*****DIVERSITY*****\n'
for epoch in epoch_recresult:
    rec_result = np.load(path + f'{epoch}_recresult_{model_name}_{dataset}.npy')
    
    ild_all = []
    
    recmodel.eval()
    with torch.no_grad():
        
        for cdd in rec_result:
        
            cdd_k = torch.from_numpy(cdd[:k])
            # (k,) np.array
            C = recmodel.embedding_item(cdd_k) # (k, dim_node)
            C_t = torch.transpose(C, 0, 1)
            _S = torch.matmul(C, C_t)
            ild = (2.0/k*(k-1))*(1-torch.sigmoid(torch.sum(_S)-_S.trace()))
    
            ild_all.append(ild)
    
    print(f'{dataset} {model_name} {epoch}: {np.mean(ild_all)}')
    _f += f'{dataset} {model_name} {epoch}: {np.mean(ild_all)}\n'

    ILD.append(np.mean(ild_all))

f.write(_f)
f.close()


