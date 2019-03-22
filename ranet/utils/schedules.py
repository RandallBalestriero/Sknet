#!/usr/bin/env python
# -*- coding: utf-8 -*-



class constant:
    def __init__(self,init_lr):
        self.lr = init_lr
        self.lrs= list()
    def update(self,valid_accu,epoch):
        self.lrs.append(self.lr)

class linear:
    def __init__(self,init_lr,step, adaptive=False):
        self.step     = step
        self.adaptive = adaptive
        self.lr       = init_lr
        self.lrs      = list()
    def update(self,valid_accu,epoch):
        if self.adaptive:
            if adaptive(valid_accu):
                self.lr  -= self.step
        else:
            self.lr  -= self.step
        self.lrs.append(self.lr)

class exponential:
    def __init__(self,init_lr,step, adaptive = False):
        self.step     = step
        self.lr       = init_lr
        self.adaptive = adaptive
        self.lrs      = list()
    def update(self,valid_accu,epoch):
        if self.adaptive:
            if adaptive(valid_accu):
                self.lr  *= self.step
        else:
            self.lr  *= self.step
        self.lr  *= self.step
        self.lrs.append(self.lr)

class stepwise:
    def __init__(self,dict_lr):
        self.dict_lr = dict_lr
        self.lr      = dict_lr[0]
        self.lrs     = list()
    def update(self,valid_accu,epoch):
        if epoch in self.dict_lr.keys():
            self.lr = self.dict_lr[epoch]
        self.lrs.append(self.lr)




def adaptive(valid_accu):
    if len(valid_accu)<10:
        return False
    if np.std(valid_loss[-10])<0.1:
        return True
    if valid_accu[-1]<valid_accu[-2] and valid_accu[-2]<valid_accu[-3]:
        return True
    return False










