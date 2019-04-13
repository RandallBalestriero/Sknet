import os
import pickle,gzip
import urllib.request
import numpy as np
import time
from . import Dataset

from ..utils import to_one_hot

def load_mnist(PATH=None):
    """Grayscale digit classification.
    The `MNIST <http://yann.lecun.com/exdb/mnist/>`_ database of handwritten 
    digits, available from this page, has a training set of 60,000 examples, 
    and a test set of 10,000 examples. It is a subset of a larger set available 
    from NIST. The digits have been size-normalized and centered in a 
    fixed-size image. It is a good database for people who want to try learning
    techniques and pattern recognition methods on real-world data while 
    spending minimal efforts on preprocessing and formatting.

    :param path: (optional, default $DATASET_PATH), the path to look for the data and
                     where the data will be downloaded if not present
    :type path: str
    """
    #init
    #Set up the configuration for data loading and data format

    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    datum_shape = (1,28,28)
    dict_init = [("n_classes",10),("name","mnist"),
                    ('classes',[str(u) for u in range(10)])]

    mnist_dataset = Dataset(**dict(dict_init))
    # Load the dataset (download if necessary) and set
    # the class attributes.
    print('Loading mnist')

    t    = time.time()

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

    # Loading the file
    f = gzip.open(PATH+'mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
    f.close()

    images = {'train_set':train_set[0].reshape((-1,1,28,28)),
            'valid_set':valid_set[0].reshape((-1,1,28,28)),
            'test_set':test_set[0].reshape((-1,1,28,28))}

    labels = {'train_set':train_set[1],
        'valid_set':valid_set[1],
        'test_set':test_set[1]}


    mnist_dataset.add_variable({'images':[images,(1,28,28),'float32'],
                                'labels':[labels,(),'int32']})

    print('Dataset mnist loaded in','{0:.2f}'.format(time.time()-t),'s.')

    return mnist_dataset

