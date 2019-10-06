import os
import numpy as np
import time
from . import Dataset


from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split





def load_path(N=1000, std=0.03, option=0, seed=None):
    X  = np.linspace(0,1,N)
    if option==0:
        # simple cosine
        X  = np.stack([X*9,np.cos(X*9)],1)
    elif option==1:
        # circle
        X -= 0.5
        X *= 2
        Y  = np.concatenate([np.sqrt(1-X[::2]**2),-np.sqrt(1-X[::2]**2)],0)
        X  = np.concatenate([X[::2],X[::2]],0)
        X  = np.stack([X,Y],1)
    elif option==2:
        # swiss roll
        t = np.linspace(0,2,N)
        X = np.stack([t*np.cos(3.14*t),t*np.sin(3.14*t)],1)
    elif option==3:
        # square
        left_side  = np.stack([np.ones(N//4)*X[0],X[::4]],1)
        right_side = np.stack([np.ones(N//4)*X[-1],X[::4]],1)
        upper_side = np.stack([X[::4],np.ones(N//4)*X[0]],1)
        lower_side = np.stack([X[::4],np.ones(N//4)*X[-1]],1)
        X          = np.concatenate([left_side,upper_side,right_side,
                                                         lower_side[::-1]],0)
    elif option == 4:
        # simple cosine
        X  = np.stack([X*9, np.random.rand(*X.shape), np.cos(X*9)],1)
    elif option == 5:
#        X = np.random.randn(N, 3)
#        X /= np.linalg.norm(X, 2, axis=1, keepdims=True)
#        X[:,0] = np.abs(X[:, 0])
        import sklearn
        X, _ = sklearn.datasets.make_swiss_roll(N,0)
    X += np.random.RandomState(seed).randn(N, X.shape[1])*std
    X -= X.mean(0, keepdims=True)
    X /= X.max(0, keepdims=True)

    X=X.astype('float32')

    dataset= Dataset()
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

def load_onesine(N):
    time = np.linspace(-1.5, 5.5, N//2)
    train_set = np.stack([time, np.cos(time)], 1)
    train_set = np.concatenate([train_set, train_set], 0)
    train_set = train_set.astype('float32')
    labels = np.zeros(N, dtype='int32')
    return train_set, labels

def load_onecircle(N):
    X = np.random.randn(N, 2)
    X /= np.sqrt((X**2).sum(1, keepdims=True))
    labels = np.zeros(N)
    X = X.astype('float32')
    labels = labels.astype('int32')
    return X, labels

def load_onesquare(N):
    time = np.linspace(0, 1, N//4).reshape((-1, 1))
    print(time.shape, np.ones(N//4).shape)
    np.hstack([time, np.ones((N//4, 1))])
    X = np.vstack([np.hstack([time, np.ones((N//4, 1))]),
                        np.hstack([time, np.zeros((N//4, 1))]),
                        np.hstack([np.zeros((N//4, 1)), time]),
                        np.hstack([np.ones((N//4, 1)), time])])
    X -= X.mean(0)
    labels = np.zeros(N)
    labels = labels.astype('int32')
    X = X.astype('float32')
    return X, labels



def load_twosine(N, factor=0.3):
    time = np.linspace(-1.5, 5.5, N//2)
    train_set = np.stack([time, np.cos(time)], 1)
    train_set = np.concatenate([train_set, train_set], 0)
    train_set[N//2:, 1] += factor
    train_set = train_set.astype('float32')
    labels = np.zeros(N, dtype='int32')
    labels[N//2:] += 1
    return train_set, labels

def load_twocircles(N, factor=0.3):
    X = np.random.randn(N, 2)
    X /= np.sqrt((X**2).sum(1, keepdims=True))
    X[:N//2] *= (1+factor)
    labels = np.concatenate([np.ones(N//2), np.zeros(N//2)])
    X = X.astype('float32')
    labels = labels.astype('int32')
    return X, labels

def load_twosquares(N, factor=0.3):
    time = np.linspace(0, 1, N//4).reshape((-1, 1))
    print(time.shape, np.ones(N//4).shape)
    np.hstack([time, np.ones((N//4, 1))])
    X = np.vstack([np.hstack([time, np.ones((N//4, 1))]),
                        np.hstack([time, np.zeros((N//4, 1))]),
                        np.hstack([np.zeros((N//4, 1)), time]),
                        np.hstack([np.ones((N//4, 1)), time])])
    X -= X.mean(0)
    X[::2] *= (1+factor)
    labels = np.ones(N)
    labels[::2] = 0
    labels = labels.astype('int32')
    X = X.astype('float32')
    return X, labels


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

