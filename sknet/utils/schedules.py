#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class constant:
    def __init__(self,init_lr):
        self.init_lr = init_lr
        self.name    = '-schedule(constant,lr='+str(init_lr)+')'
    def init(self):
        self.lr  = self.init_lr+0
        self.lrs = list()
    def update(self,valid_accu,epoch):
        self.lrs.append(self.lr)

class linear:
    def __init__(self,init_lr,step, adaptive=False):
        self.step     = step
        self.adaptive = adaptive
        self.init_lr  = init_lr
        self.name     = '-schedule(linear,lr='+str(init_lr)\
                    +',step='+str(step)+',adaptive='+str(adaptive)+')'
    def init(self):
        self.lr  = self.init_lr+0
        self.lrs = list()
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
        self.init_lr  = init_lr
        self.adaptive = adaptive
        self.name = '-schedule(exponential,lr='+str(init_lr)\
                +',step='+str(step)+',adaptive='+str(adaptive)+')'
    def init(self):
        self.lr  = self.init_lr+0
        self.lrs = list()
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
        self.name    = '-schedule(stepwise,dict='+str(dict_lr).replace(' ','')+')'
    def init(self):
        self.lr  = self.dict_lr[0]+0
        self.lrs = list()
    def update(self,valid_accu,epoch):
        if epoch in self.dict_lr.keys():
            self.lr = self.dict_lr[epoch]
        self.lrs.append(self.lr)




def adaptive(valid_accu,patience=5):
    """
    Reduce learning rate when a metric has stopped improving. 
    Models often benefit from reducing the learning rate by a 
    factor of 2-10 once learning stagnates. This scheduler reads 
    a metrics quantity and if no improvement is seen for a ‘patience’ 
    number of epochs, the learning rate is reduced.
    """
    if len(valid_accu)<10:
        return False
    if np.std(valid_accu[-10])<0.1:
        return True
    if valid_accu[-1]<valid_accu[-2] and valid_accu[-2]<valid_accu[-3]:
        return True
    return False










