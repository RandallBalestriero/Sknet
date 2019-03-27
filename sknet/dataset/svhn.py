import scipy.io as sio
import os
import pickle,gzip
import urllib.request
import numpy as np
import time



class svhn:
    """ The `SVHN <http://ufldl.stanford.edu/housenumbers/>`_
    dataset is a real-world 
    image dataset for developing machine learning and object 
    recognition algorithms with minimal requirement on data 
    preprocessing and formatting. It can be seen as similar in flavor 
    to MNIST (e.g., the images are of small cropped digits), but 
    incorporates an order of magnitude more labeled data (over 600,000 
    digit images) and comes from a significantly harder, unsolved, 
    real world problem (recognizing digits and numbers in natural 
    scene images). SVHN is obtained from house numbers in Google 
    Street View images. 
    """
    def __init__(self,data_format='NCHW'):
        self.data_format     = data_format
        self.given_test_set  = True
        self.given_valid_set = False
        self.given_unlabeled = False
        self.n_classes       = 10
        self.classes         = dict([[u,str(u)] for u in range(10)])
        self.name            = 'svhn'
    def load(self):

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

        # Check formatting
        if data_format=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
            self.datum_shape = (32,32,3)
        else:
            self.datum_shape = (3,32,32)

        self.train_set = train_set
        self.test_set  = test_set

        print('Dataset SVHN loaded in','{0:.2f}'.format(time.time()-t),'s.')












