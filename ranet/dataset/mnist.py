import os
import pickle,gzip
import urllib.request
import numpy as np

def load(data_format='NCHW'):

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
    # Check formatting
    if data_format=='NHWC':
        train_set[0] = np.transpose(train_set[0],[0,2,3,1])
        test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
        valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
    return train_set, valid_set, test_set




