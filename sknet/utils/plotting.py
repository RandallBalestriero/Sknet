#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as mp
import numpy as np


def imshow(x,cmap='jet'):
    xmin = x.min()
    xmax = x.max()
    # Ensure that color channel is last
    if x.shape[0]==3:
        x = x.transpose([1,2,0])
    # Squeeze to remove the case of single channel
    mp.imshow((np.squeeze(x)-xmin)/(xmax-xmin),cmap=cmap,aspect='auto',interpolation='kaiser')
    mp.xticks([])
    mp.yticks([])

