'''
input:
    model weight file
output:
    test result
'''

import torch
from tqdm import tqdm
import numpy as np
import random
import model
import config
from metrics import compute_metrics
from data_pipeline import generate_batch_test

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(63)
print('random seed set...')

Model = model.MyModel(**config.model_args['mymodel'])
date = '01111211'#'01132333'#'01111211'
dataset = 'steam'#'movielens'#'steam'

def compute_acc(epoch_model):

    weight_file = f'./weights/{date}/mymodel_{dataset}_{epoch_model}.pth.tar'

    Model.load_state_dict(torch.load(weight_file))
    device = torch.device('cuda')
    Model.to(device)

    Model.eval()

    batch_generator_test = generate_batch_test(path=f'./data/{dataset}', batch_user = 1, device = torch.device('cuda'))

    with torch.no_grad():

        accs_all = []
        for user, candidate in batch_generator_test:
            scores = Model.compute_rating(user, candidate).cpu().numpy()
            label_origin = candidate.label
            
            accs_5 = compute_metrics(scores, label_origin, 5)
            accs_10 = compute_metrics(scores, label_origin, 10)
            accs_20 = compute_metrics(scores, label_origin, 20)

            accs_all.append([*accs_5, *accs_10, *accs_20])
            cdd_sorted = candidate.cdd.squeeze().cpu().numpy()[np.argsort(scores)[::-1]]
        aver_accs = np.mean(np.array(accs_all), axis = 0)
        hit_5, mrr_5, hit_10, mrr_10, hit_20, mrr_20 = aver_accs

    result = f"{{\'hit@5\':{hit_5:.3e}, \'mrr@5\':{mrr_5:.3e}, \'hit@10\':{hit_10:.3e}, \'mrr@10\':{mrr_10:.3e}, \'hit@20\':{hit_20:.3e}, \'mrr@20\':{mrr_20:.3e}}}"
    return eval(result)

if __name__ == '__main__':

    for epoch_model in range(30):
        acc = compute_acc(epoch_model=epoch_model)
        print(acc)

