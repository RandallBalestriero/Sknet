import os
import numpy as np
import time
from . import Dataset


from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split

def load_mini():
    """Grayscale digit classification.
    The `MNIST <http://yann.lecun.com/exdb/mnist/>`_ database of handwritten 
    digits, available from this page, has a training set of 60,000 examples, 
    and a test set of 10,000 examples. It is a subset of a larger set available 
    from NIST. The digits have been size-normalized and centered in a 
    fixed-size image. It is a good database for people who want to try learning
    techniques and pattern recognition methods on real-world data while 
    spending minimal efforts on preprocessing and formatting.

    :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
    :type data_format: 'NCHW' or 'NHWC'
    :param path: (optional, default $DATASET_PATH), the path to look for the data and
                     where the data will be downloaded if not present
    :type path: str
    """

    X,y   = make_moons(10000,noise=0.035,random_state=20)
    x_,y_ = make_circles(10000,noise=0.02,random_state=20)
    x_[:,1]+= 2.
    y_   += 2
    X     = np.concatenate([X,x_],axis=0)
    y     = np.concatenate([y,y_])
    X    -= X.mean(0,keepdims=True)
    X    /= X.max(0,keepdims=True)

    X=X.astype('float32')
    y=y.astype('int32')

    dict_init = [("datum_shape",(2,)),("n_classes",4),
                    ("name","mini"),("data_format","D"),
                    ('classes',[str(u) for u in range(4)])]

    dataset= Dataset(**dict(dict_init))

    images = {'train_set':X}

    labels = {'train_set':y}

    dataset.add_variable({'images':images,'labels':labels})

    return dataset

