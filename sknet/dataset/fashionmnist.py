import os
import gzip
import urllib.request
import numpy as np
import time





class fashionmnist:
    def __init__(self):
        pass
    def load(self,data_format='NCHW',seed=None):

        self.name          = 'fashionmnist'
        self.data_format   = data_format
        self.classes       = 10

        t = time.time()

        PATH = os.environ['DATASET_PATH']

        if not os.path.isdir(PATH+'fashionmnist'):
            print('Creating Directory')
            os.mkdir(PATH+'fashionmnist')

        if not os.path.exists(PATH+'fashionmnist/train-images-idx3-ubyte.gz'):
            print('Downloading FASHION-MNIST Train Images')
            urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',PATH+'fashionmnist/train-images-idx3-ubyte.gz')

        if not os.path.exists(PATH+'fashionmnist/train-labels-idx1-ubyte.gz'):
            print('Downloading FASHION-MNIST Train Labels')
            urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',PATH+'fashionmnist/train-labels-idx1-ubyte.gz')

        if not os.path.exists(PATH+'fashionmnist/t10k-images-idx3-ubyte.gz'):
            print('Downloading FASHION-MNIST Test Images')
            urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',PATH+'fashionmnist/t10k-images-idx3-ubyte.gz')

        if not os.path.exists(PATH+'fashionmnist/t10k-labels-idx1-ubyte.gz'):
            print('Downloading FASHION-MNIST Test Labels')
            urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',PATH+'fashionmnist/t10k-labels-idx1-ubyte.gz')

        # Loading the file
        print('Loading FASHION-MNIST')
        train_set = [[],[]]
        test_set  = [[],[]]

        with gzip.open(PATH+'fashionmnist/train-labels-idx1-ubyte.gz', 'rb') as lbpath:
            train_set[1] = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(PATH+'fashionmnist/train-images-idx3-ubyte.gz', 'rb') as lbpath:
            train_set[0] = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16).reshape((len(train_set[1]),1,28,28))

        with gzip.open(PATH+'fashionmnist/t10k-labels-idx1-ubyte.gz', 'rb') as lbpath:
            test_set[1] = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(PATH+'fashionmnist/t10k-images-idx3-ubyte.gz', 'rb') as lbpath:
            test_set[0] = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16).reshape((len(test_set[1]),1,28,28))

        # Compute a Valid Set
        random_indices = np.random.RandomState(seed=seed).permutation(train_set[0].shape[0])
        train_indices  = random_indices[:int(train_set[0].shape[0]*0.9)]
        valid_indices  = random_indices[int(train_set[0].shape[0]*0.9):]
        valid_set      = [train_set[0][valid_indices],train_set[1][valid_indices]]
        train_set      = [train_set[0][train_indices],train_set[1][train_indices]]

        # Check Formatting
        if data_format=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
            valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
            self.image_shape = (28,28,1)
        else:
            self.image_shape = (1,28,28)

        print('Dataset FASHION-MNIST loaded in','{0:.2f}'.format(time.time()-t),'s.')

        return train_set, valid_set, test_set



