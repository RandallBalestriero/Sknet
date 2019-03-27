import urllib.request
import time
import sys
import os, sys, tarfile
import numpy as np
import matplotlib.pyplot as plt
    

class stl10:
    """ 
    The `STL-10 <https://cs.stanford.edu/~acoates/stl10/>`_ dataset is an image 
    recognition dataset for developing unsupervised feature learning, 
    deep learning, self-taught learning algorithms. It is inspired by the 
    `CIFAR-10 <http://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset but with 
    some modifications. In particular, each class has fewer labeled 
    training examples than in CIFAR-10, but a very 
    large set of unlabeled examples is provided to learn image models prior 
    to supervised training. The primary challenge is to make use of the 
    unlabeled data (which comes from a similar but different distribution from 
    the labeled data) to build a useful prior. We also expect that the higher 
    resolution of this dataset (96x96) will make it a challenging benchmark 
    for developing more scalable unsupervised learning methods.
    """
    def __init__(self,data_format='NCHW'):
        self.data_format     = data_format
        self.given_test_set  = True
        self.given_valid_set = False
        self.given_unlabeled = True
        self.n_classes       = 10
        self.name            = 'stl10'
        self.classes         = {0:"airplane", 
                                1:"bird", 
                                2:"car", 
                                3:"cat", 
                                4:"deer", 
                                5:"dog", 
                                6:"horse", 
                                7:"monkey", 
                                8:"ship", 
                                9:"truck"}
    def load(self,seed=None):
        t    = time.time()
        PATH = os.environ['DATASET_PATH']

        # Check if directory exists
        if not os.path.isdir(PATH+'stl10'):
            print('Creating Directory')
            os.mkdir(PATH+'stl10')

        # Check if data file exists
        if not os.path.exists(PATH+'stl10/stl10_binary.tar.gz'):
            print('Downloading Data')
            url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
            urllib.request.urlretrieve(url,PATH+'stl10/stl10_binary.pkl.gz')  

        # Loading Dataset
        print('Loading STL-10')
        file_ = tarfile.open(PATH+'stl10/stl10_binary.pkl.gz', 'r:gz')
        # loading test label
        read_file = file_.extractfile('stl10_binary/test_y.bin').read()
        test_y = np.frombuffer(io.BytesIO(label_file).read(), dtype=np.uint8)
        # loading train label
        read_file = file_.extractfile('stl10_binary/train_y.bin').read()
        train_y = np.frombuffer(io.BytesIO(label_file).read(), dtype=np.uint8)
        # load test images
        read_file = file_.extractfile('stl10_binary/test_X.bin').read()
        test_X = np.frombuffer(io.BytesIO(label_file).read(), dtype=np.uint8).reshape((-1,3,96,96))
        # load train images
        read_file = file_.extractfile('stl10_binary/train_X.bin').read()
        train_X = np.frombuffer(io.BytesIO(label_file).read(), dtype=np.uint8).reshape((-1,3,96,96))
        # load unlabelled images
        read_file = file_.extractfile('stl10_binary/unlabeled_X.bin').read()
        unlabeled_X = np.frombuffer(io.BytesIO(label_file).read(), dtype=np.uint8).reshape((-1,3,96,96))

        # Check formatting
        if data_format=='NHWC':
            train_X     = np.transpose(train_X,[0,2,3,1])
            test_X      = np.transpose(test_X,[0,2,3,1])
            unlabeled_X = np.transpose(unlabeled_X,[0,2,3,1])
            self.datum_shape = (96,96,3)
        else:
            self.datum_shape = (3,96,96)
            
        self.train_set = [train_X,train_y]
        self.test_set  = [test_X,test_y]
        self.unlabeled = [unlabeled_X]
        print('Dataset STL10 loaded in','{0:.2f}'.format(time.time()-t),'s.')





