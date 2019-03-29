import os
import pickle,gzip
import urllib.request
import numpy as np
import time



class mnist(dict):
    """Grayscale digit classification.
    The `MNIST <http://yann.lecun.com/exdb/mnist/>`_ database of handwritten 
    digits, available from this page, has a training set of 60,000 examples, 
    and a test set of 10,000 examples. It is a subset of a larger set available 
    from NIST. The digits have been size-normalized and centered in a 
    fixed-size image. It is a good database for people who want to try learning
    techniques and pattern recognition methods on real-world data while 
    spending minimal efforts on preprocessing and formatting.
    """
    def __init__(self,data_format='NCHW',path=None):
        """Set up the configuration for data loading and data format

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :param path: (optional, default $DATASET_PATH), the path to look for the data and 
                     where the data will be downloaded if not present
        :type path: str
        """
        if path is None:
            path = os.environ['DATASET_PATH']
        if data_format=='NCHW':
            datum_shape = (1,28,28)
        else:
            datum_shape = (28,28,1)
        dict_init = [("train_set",None),("test_set",None),("valid_set",None),
                    ("datum_shape",datum_shape),("n_classes",10),
                    ("n_channels",1),("spatial_shape",(28,28)),
                    ("path",path),("data_format",data_format),("name","mnist"),
                    ('classes',[str(u) for u in range(10)])]
        super().__init__(dict_init)
    def load(self):
        """Load the dataset (download if necessary) and set
        the class attributes.
        """
        print('Loading mnist')

        t    = time.time()
        PATH = self['path']

        # Check if directory exists
        if not os.path.isdir(PATH+'mnist'):
            print('Creating mnist Directory')
            os.mkdir(PATH+'mnist')

        # Check if file exists
        if not os.path.exists(PATH+'mnist/mnist.pkl.gz'):
            print('\tDownloading mnist Dataset...')
            td  = time.time()
            url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
            urllib.request.urlretrieve(url,PATH+'mnist/mnist.pkl.gz')
            print("\tDone in {:.2f}".format(time.time()-td))

        print('\tOpening mnist')
        # Loading the file
        f = gzip.open(PATH+'mnist/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
        f.close()
        train_set = [train_set[0].reshape((-1,1,28,28)),train_set[1]]
        test_set  = [test_set[0].reshape((-1,1,28,28)),test_set[1]]
        valid_set = [valid_set[0].reshape((-1,1,28,28)),valid_set[1]]
        # Check formatting
        if self["data_format"]=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
            valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
    
        self['train_set'] = train_set
        self['test_set']  = test_set
        self['valid_set'] = valid_set

        print('Dataset mnist loaded in','{0:.2f}'.format(time.time()-t),'s.')

