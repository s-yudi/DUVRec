
import multiprocessing
import utils
from tqdm import tqdm
import numpy as np
import torch

from data_pipeline import generate_batch_train, generate_batch_test
from metrics import compute_metrics
import config

CORES = multiprocessing.cpu_count() // 2

def BPR_train(path, recmodel, loss_class, batch_size, device, **kwargs):
    Recmodel = recmodel
    Recmodel.train()
    bpr: BPRLoss = loss_class

    with open(path + '/train.txt', 'r') as f:
        train_size = len(f.readlines())

    total_batch = train_size//batch_size + 1
    aver_loss = 0.
    
    batch_generator = generate_batch_train(path=path + '/', batch_size=batch_size, device = device)

    for batch_i in tqdm(range(total_batch), total=float("inf")):
        batch_users, batch_candidates = next(batch_generator)
        cri = bpr.stageOne(batch_users, batch_candidates)
        aver_loss += cri
        #raise Exception('break')
        
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"

def Test(path, recmodel, batch_user, device, epoch, k=20, **kwargs):
    
    Recmodel = recmodel
    Recmodel.eval()
    
    batch_generator_test = generate_batch_test(path=path, device = device)

    with torch.no_grad():
    
        accs_all = []
        rec_result_all = []

        for user, candidate in tqdm(batch_generator_test):

            scores = Recmodel.compute_rating(user, candidate).cpu().numpy()

            label_origin = candidate.label
            accs_10 = compute_metrics(scores, label_origin, 10)
            accs_5 = compute_metrics(scores, label_origin, 5)

            accs_all.append(accs_10+accs_5)

            cdd_sorted = candidate.cdd.squeeze().cpu().numpy()[np.argsort(scores)[::-1]]
            rec_result_all.append(cdd_sorted)

        aver_accs = np.mean(np.array(accs_all), axis = 0)
        hit_10, mrr_10, hit_5, mrr_5 = aver_accs
    
    rec_result_all = np.array(rec_result_all)
    np.save(config.info['weight_folder']+f"{epoch}_recresult_{config.model_choice}_{config.dataset_choice}.npy", rec_result_all)

    result = f"{{\'hit@{10}\':{hit_10:.3e}, \'mrr@{10}\':{mrr_10:.3e},\'hit@{5}\':{hit_5:.3e}, \'mrr@{5}\':{mrr_5:.3e}}}"
    print(result)

    return eval(result)

if __name__=='__main__':

    from model import MyModel
    import loss

    config = {
        'batch_size': 1024,
        'decay': 1e-4,
        'lr': 1e-3
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Recmodel = MyModel(dim_node=12,\
                dim_hidden=64, \
                n_assign=5, \
                n_heads=2, \
                num_items_all=11667 \
                ).to(device)
    #Recmodel = Recmodel.to(device)

    '''
    bpr = loss.BPRLoss(Recmodel, config)

    output_information = BPR_train('./data/steam', Recmodel, bpr, 1024)
    '''

    print(Test(path='./data/steam', recmodel=Recmodel, device=device))


