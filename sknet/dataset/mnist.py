import os
import pickle,gzip
import urllib.request
import numpy as np
import time



class mnist:
    """
    The `MNIST <http://yann.lecun.com/exdb/mnist/>`_ database of handwritten 
    digits, available from this page, has a training set of 60,000 examples, 
    and a test set of 10,000 examples. It is a subset of a larger set available 
    from NIST. The digits have been size-normalized and centered in a 
    fixed-size image. It is a good database for people who want to try learning
    techniques and pattern recognition methods on real-world data while 
    spending minimal efforts on preprocessing and formatting.
    """
    def __init__(self,data_format='NCHW'):
        self.data_format     = data_format
        self.given_test_set  = True
        self.given_valid_set = True
        self.given_unlabeled = False
        self.n_classes       = 10
        self.name            = 'mnist'
        self.classes         = {0:"0",
                                1:"1",
                                2:"2",
                                3:"3",
                                4:"4",
                                5:"5",
                                6:"6",
                                7:"7",
                                8:"8",
                                9:"9"}
    def load(self):
        t = time.time()
        PATH = os.environ['DATASET_PATH']

        # Check if directory exists
        if not os.path.isdir(PATH+'mnist'):
            print('Creating Directory')
            os.mkdir(PATH+'mnist')

        # Check if file exists
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
            self.datum_shape = (28,28,1)
        else:
            self.datum_shape = (1,28,28)
    
        self.train_set = train_set
        self.test_set  = test_set
        self.valid_set = valid_set
        print('Dataset MNIST loaded in','{0:.2f}'.format(time.time()-t),'s.')





