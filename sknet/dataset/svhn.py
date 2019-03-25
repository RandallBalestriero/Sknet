import scipy.io as sio
import os
import pickle,gzip
import urllib.request
import numpy as np
import time



class svhn:
    def __init__(self):
        pass
    def load(self,data_format='NCHW',seed=None):

        self.name          = 'svhn'
        self.data_format   = data_format
        self.classes       = 10

        t = time.time()

        PATH = os.environ['DATASET_PATH']

        if not os.path.isdir(PATH+'svhn'):
            os.mkdir(PATH+'svhn')
            print('Creating Data Directory')

        if not os.path.exists(PATH+'svhn/train_32x32.mat'):
            print('Downloading Training File')
            url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
            urllib.request.urlretrieve(url,PATH+'svhn/train_32x32.mat')

        if not os.path.exists(PATH+'svhn/test_32x32.mat'):
            print('Downloading Testing File')
            url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
            urllib.request.urlretrieve(url,PATH+'svhn/test_32x32.mat')

        print('Loading SVHN')
        train_set = sio.loadmat(PATH+'svhn/train_32x32.mat')
        train_set = [train_set['X'].transpose([3,2,0,1]).astype('float32'),
                    train_set['y'].astype('int32')-1]
        test_set  = sio.loadmat(PATH+'svhn/test_32x32.mat')
        test_set  = [test_set['X'].transpose([3,2,0,1]).astype('float32'),
                    test_set['y'].astype('int32')-1]

        # Compute a Valid Set
        random_indices = np.random.RandomState(seed=seed).permutation(train_set[0].shape[0])
        train_indices  = random_indices[:int(train_set[0].shape[0]*0.9)]
        valid_indices  = random_indices[int(train_set[0].shape[0]*0.9):]
        valid_set      = [train_set[0][valid_indices],train_set[1][valid_indices]]
        train_set      = [train_set[0][train_indices],train_set[1][train_indices]]
        # Check formatting
        if data_format=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
            valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
            self.image_shape = (32,32,3)
        else:
            self.image_shape = (3,32,32)

        print('Dataset SVHN loaded in','{0:.2f}'.format(time.time()-t),'s.')

        return train_set, valid_set, test_set











