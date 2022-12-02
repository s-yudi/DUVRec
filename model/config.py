
'''
只提供配置，不提供选择
'''

import torch
import os
cur_path = os.path.split(os.path.realpath(__file__))[0]
'''note
时间间隔缩放到[0,10]区间内
'''
#----------------------------------------------------------------------
info = {
    'device'       : torch.device('cuda' if torch.cuda.is_available else 'cpu'),
    'weight_folder': cur_path + '/../weights/',
    'cwd'          : cur_path + '/../'
}
#info['device'] = torch.device('cpu')

#----------------------------------------------------------------------
dataset_choice = 'steam'
# steam, movielens, toys
dataset = {
    'steam':{
        'path'             : info['cwd']+'data/steam/',
        'onesample_size'   : 12,
        'num_items_all'    : 11667,
        'num_all'          : 11667+22,
        'max_position_span': 60,
        'n_assign' : 5
    },
    'movielens':{ 
        'path'             : info['cwd']+'data/movielens/',
        'onesample_size'   : 12,
        'num_items_all'    : 3390,
        'num_all'          : 3390+20,
        'max_position_span': 60,
        'n_assign' : 5
    },
    'toys':{
        'path'             : info['cwd']+'data/toys/',
        'onesample_size'   : 12,
        'num_items_all'    :624792 ,
        'num_all'          : 624792+665,
        'max_position_span': 60,
        'n_assign' : 7
    }
}

#----------------------------------------------------------------------
model_choice = 'mymodel'
# mymodel, flat_attention
model_args = {
    'mymodel':{
        'dim_node'     : 16,#dataset[dataset_choice]['onesample_size'], #64
        'num_items_all': dataset[dataset_choice]['num_items_all'],
        'num_all'      : dataset[dataset_choice]['num_all'],
        'dim_hidden'   : 32,#64,
        'n_assign'     : dataset[dataset_choice]['n_assign'],
        'n_heads'      : 2,
        'device'       : info['device']
    }
}

loss = {
    'decay_reg': 1e-3,
    'decay_ent': 1e-4,
    'decay_con': 1e-3,
    'lr'   : 1e-3
}

train = {
    'path'         : info['cwd']+f'data/{dataset_choice}',
    'num_epoch'    : 10000,
    'history_size' : dataset[dataset_choice]['onesample_size'],
    'batch_size'   : 1024,
    'device'       : info['device'],
    'tr_per_te'    : 1,
    'num_earlyStop': 10
}

test = {
    'path'         : info['cwd']+f'data/{dataset_choice}',
    'batch_user'   : 1,
    'device'       : info['device'],
    'k'            : 10
}

#-------------------------------script---------------------------------

'''
# dataset['max_position_span']
max_position_span = -float('inf')
for filename in ['train.txt', 'test.txt']:
    with open(dataset[dataset_choice]['path'] + filename, 'r') as f:
        for row in f:
            row = eval(row)[1]  
            span = (row[-1]-row[0])/(3600*24)
            max_position_span = max(span, max_position_span)
dataset[dataset_choice]['max_position_span'] = int(max_position_span+1)
'''

# weight_folder
# clear empty folder

for item in os.listdir(info['weight_folder']):
    item_path = os.path.join(info['weight_folder'], item)
    if os.path.isdir(item_path) and len(os.listdir(item_path))==0:
        os.rmdir(item_path)
# prepare weight_folder
from datetime import datetime
info['weight_folder'] += datetime.now().strftime('%m%d%H%M')+dataset_choice+''+'/'
if not os.path.exists(info['weight_folder']):
    os.mkdir(info['weight_folder'])

