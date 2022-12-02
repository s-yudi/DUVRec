
import torch
from torch import nn, optim
import config

class BPRLoss:
    def __init__(self, 
                 recmodel, 
                 decay_reg, decay_ent, decay_con, lr, **kwargs):
        self.model = recmodel
        self.decay_reg = decay_reg
        self.decay_ent = decay_ent
        self.decay_con = decay_con
        self.lr = lr
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        
    def stageOne(self, users, candidates):
        
        loss, reg_loss, entropy_loss, con_loss = self.model.bpr_loss(users, candidates)

        if torch.rand(1)[0]>0.8:
            print(f'loss:{loss:.2f}, reg_loss:{reg_loss:.2f}, entropy_loss:{entropy_loss:.2f}, con_loss:{con_loss:.2f}')

        '''
        print('*'*10)
        print(loss)
        print(reg_loss)
        print(entropy_loss)
        print(con_loss)
        print('*'*10)
        '''
        reg_loss = reg_loss*self.decay_reg
        entropy_loss = entropy_loss*self.decay_ent
        con_loss = con_loss*self.decay_con
        loss = loss + reg_loss + entropy_loss + con_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.cpu().item()
