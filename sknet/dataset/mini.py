import os
import numpy as np
import time
from . import Dataset


from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split

def load_mini():
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
                    ("name","mini"),('classes',[str(u) for u in range(4)])]

    dataset= Dataset(**dict(dict_init))

    images = {'train_set':X}

    labels = {'train_set':y}

    dataset.add_variable({'images':images,'labels':labels})

    return dataset

def load_chirp2D(N,seed=0):
    X1,X2 = np.meshgrid(np.linspace(0,4,N),np.linspace(0,4,N))
    np.random.seed(seed)
    M     = np.array([[1.4,-0.4],
                    [-0.4,0.6]])
    X     = np.stack([X1.flatten(),X2.flatten()],1)
    y     = np.sin((X*np.dot(X,M)).sum(1))
    y    -= y.mean()
    y    /= y.max()

    X=X.astype('float32')
    y=y.astype('float32')

    dict_init = [("datum_shape",(2,)),("name","chirp2D")]

    dataset= Dataset(**dict(dict_init))

    images = {'train_set':X}

    labels = {'train_set':y}

    dataset.add_variable({'input':images,'output':labels})

    return dataset

