import os
import pickle,gzip
import urllib.request
import numpy as np

def load(data_format='NCHW'):

    PATH = os.environ['DATASET_PATH']

    if not os.path.isdir(PATH+'mnist'):
        os.mkdir(PATH+'mnist')

    if not os.path.exists(PATH+'mnist/mnist.pkl.gz'):
        urllib.request.urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz',PATH+'mnist/mnist.pkl.gz')  
    # Loading the file
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # Check formatting
    if data_format=='NHWC':
        train_set[0] = np.transpose(train_set[0],[0,2,3,1])
        test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
        valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
    return train_set, valid_set, test_set




