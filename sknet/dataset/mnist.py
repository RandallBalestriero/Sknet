import os
import pickle,gzip
import urllib.request
import numpy as np
import time



class mnist:
    def __init__(self):
        pass
    def load(self,data_format='NCHW',seed=None):
        self.data_format = data_format
        self.classes      = 10
        self.name        = 'mnist'
        t = time.time()
        PATH = os.environ['DATASET_PATH']
        if not os.path.isdir(PATH+'mnist'):
            print('Creating Directory')
            os.mkdir(PATH+'mnist')

        if not os.path.exists(PATH+'mnist/mnist.pkl.gz'):
            print('Downloading Data')
            urllib.request.urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz',PATH+'mnist/mnist.pkl.gz')  

        print('Loading MNIST')
        # Loading the file
        f = gzip.open(PATH+'mnist/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
        f.close()
        train_set = [train_set[0].reshape((-1,1,28,28)),train_set[1]]
        test_set  = [test_set[0].reshape((-1,1,28,28)),test_set[1]]
        valid_set = [valid_set[0].reshape((-1,1,28,28)),valid_set[1]]
        # Check formatting
        if data_format=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
            valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
            self.image_shape = (28,28,1)
        else:
            self.image_shape = (1,28,28)
            
        print('Dataset MNIST loaded in','{0:.2f}'.format(time.time()-t),'s.')

        return train_set, valid_set, test_set




