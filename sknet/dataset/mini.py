import os
import numpy as np
import time
from . import Dataset


from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split





def load_path(N=1000, std=0.1, opt=0):
    X  = np.random.rand(N)*9
    X  = np.stack([X,np.cos(X)+np.random.randn(N)*std],1)
    X -= X.mean(0,keepdims=True)
    X /= X.max(0,keepdims=True)

    X=X.astype('float32')

    dict_init = [("datum_shape",(2,)),("n_classes",1),
                    ("name","path"),('classes',[str(u) for u in range(1)])]

    dataset= Dataset(**dict(dict_init))
    dataset['inputs/train_set']=X
    return dataset


def load_mini(N=1000):
    X,y   = make_moons(N,noise=0.035,random_state=20)
    x_,y_ = make_circles(N,noise=0.02,random_state=20)
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
    dataset['inputs/train_set']=X
    dataset['outputs/train_set']=y

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

