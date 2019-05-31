#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as mp
import numpy as np


def imshow(image, interpolation='kaiser', cmap='jet'):
    xmin = image.min()
    xmax = image.max()
    # Ensure that color channel is last
    if image.shape[0] == 3:
        image = image.transpose([1, 2, 0])
    # Squeeze to remove the case of single channel
    mp.imshow((np.squeeze(image)-xmin)/(xmax-xmin), cmap=cmap, aspect='auto',
              interpolation=interpolation)
    mp.xticks([])
    mp.yticks([])
