import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import numpy as np

from tqdm import tqdm_notebook, tnrange, trange

def to_np(v):
    if isinstance(v, Variable): v=v.data
    return v.cpu().numpy()


def to_gpu(x, *args, **kwargs):
    if isinstance(x,list):
      return [x_.cuda(*args, **kwargs) if torch.cuda.is_available() else x_ for x_ in x]
    else:
      return x.cuda(*args, **kwargs) if torch.cuda.is_available() else x


class MixedInputModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                 y_range=None, use_bn=False):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c,s in emb_szs])
        for emb in self.embs: 
            self.emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)

        szs = [n_emb+n_cont] + szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: 
            nn.init.kaiming_normal_(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal_(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn,self.y_range = use_bn,y_range

    #@staticmethod
    def emb_init(self,x):
        x = x.weight.data
        sc = 2/(x.size(1)+1)
        x.uniform_(-sc,sc)    
        
    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
        x = torch.cat(x, 1)
        x2 = self.bn(x_cont)
        x = self.emb_drop(x)
        x = torch.cat([x, x2], 1)
        
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)
        x = self.outp(x)
        if self.y_range:
            x = F.sigmoid(x)
            x = x*(self.y_range[1] - self.y_range[0])
            x = x+self.y_range[0]
        return x

class Stepper():
    def __init__(self, m, opt, crit, clip=0, reg_fn=None):
        self.m,self.opt,self.crit,self.clip,self.reg_fn = m,opt,crit,clip,reg_fn
        self.reset(True)

    def reset(self, train=True):
        #if train: apply_leaf(self.m, set_train_mode)
        self.m.eval()
        #if hasattr(self.m, 'reset'): self.m.reset()

    def step(self, xs, y):
        xtra = []
        output = self.m(*xs)
        if isinstance(output,(tuple,list)): output,*xtra = output
        self.opt.zero_grad()
        loss = raw_loss = self.crit(output, y)
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        #if self.clip:   # Gradient clipping
        #    nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        self.opt.step()
        return raw_loss.item() #data[0]

    def evaluate(self, xs, y):
        preds = self.m(*xs)
        if isinstance(preds,(tuple,list)): preds=preds[0]
        return preds, self.crit(preds,y)

def validate(stepper, dl, metrics):
    loss,res = [],[]
    stepper.reset(False)
    for (*x,y) in iter(dl):
        preds,l = stepper.evaluate(to_gpu(x), to_gpu(y))
        loss.append(to_np(l))
        res.append([f(to_np(preds),to_np(y)) for f in metrics])
    return [np.mean(loss)] + list(np.mean(np.stack(res),0))


class Learner():
    def __init__(self, data, model, opt_fn=None, metrics=None):
        self.data, self.model ,self.metrics = data, model, metrics
        self.opt_fn = opt_fn #or SGD_Momentum(0.9)
        self.crit,self.reg_fn = F.mse_loss,None

    def fit(self, lr, epochs):
        """ Fits a model

        Arguments:
           model (model): any pytorch module
               net = to_gpu(net)
           data (ModelData): see ModelData class and subclasses
           opt: optimizer. Example: opt=optim.Adam(net.parameters())
           epochs(int): number of epochs
           crit: loss function to optimize. Example: F.cross_entropy
           
        """
        opt = self.opt_fn(self.model.parameters(), lr)
        stepper = Stepper(self.model, opt, self.crit)
        #metrics = self.metrics or []
        avg_mom = 0.98
        batch_num = 0 
        avg_loss = 0

        for epoch in trange(epochs, desc='Epoch'):
            stepper.reset(True)
            #t = tqdm(iter(self.data.trn_dl), leave=False, total=len(self.data.trn_dl))
            # t = iter(self.data.trn_dl)
            for (*x,y) in self.data.trn_dl:
                batch_num += 1
            
                loss = stepper.step(to_gpu(x),to_gpu(y))
                avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
                debias_loss = avg_loss / (1 - avg_mom**batch_num)
                #t.set_postfix(loss=debias_loss)

                stop=False
                if stop: return
            print('loss:', debias_loss)
            vals = validate(stepper, self.data.val_dl, self.metrics)
            print(np.round([epoch, debias_loss] + vals, 6))
            stop=False
            if stop: break
                
    
    def predict(self, is_test=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        self.model.eval()
        #if hasattr(m, 'reset'): m.reset()
        with torch.no_grad():
          res = []
          for *x,y in dl: 
            res.append([self.model(*to_gpu(x)),y])
          preda,targa = zip(*res)
        return to_np(torch.cat(preda)), to_np(torch.cat(targa))

