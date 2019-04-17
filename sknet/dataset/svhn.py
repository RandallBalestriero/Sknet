import scipy.io as sio
import os
import pickle,gzip
import urllib.request
import numpy as np
import time

from . import Dataset

from ..utils import to_one_hot, DownloadProgressBar


def load_svhn(PATH=None):
    """Street number classification.
    The `SVHN <http://ufldl.stanford.edu/housenumbers/>`_
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

    :param path: (optional, default $DATASET_PATH), the path to look for the data and 
                 where the data will be downloaded if not present
    :type path: str
    """
    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    dict_init = [("n_classes",10),("path",PATH),("name","svhn"),
                ("classes",[str(1+u) for u in range(10)])]
    dataset = Dataset(**dict(dict_init))

    # Load the dataset (download if necessary) and set
    # the class attributess.
    print('Loading svhn')

    t = time.time()

    if not os.path.isdir(PATH+'svhn'):
        os.mkdir(PATH+'svhn')
        print('\tCreating svhn Directory')

    if not os.path.exists(PATH+'svhn/train_32x32.mat'):
        url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Downloading train set') as t:
            urllib.request.urlretrieve(url,PATH+'svhn/train_32x32.mat')

    if not os.path.exists(PATH+'svhn/test_32x32.mat'):
        url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Downloading test set') as t:
            urllib.request.urlretrieve(url,PATH+'svhn/test_32x32.mat')

    train_set = sio.loadmat(PATH+'svhn/train_32x32.mat')
    train_set = [train_set['X'].transpose([3,2,0,1]).astype('float32'),
                    np.squeeze(train_set['y']).astype('int32')-1]
    test_set  = sio.loadmat(PATH+'svhn/test_32x32.mat')
    test_set  = [test_set['X'].transpose([3,2,0,1]).astype('float32'),
                    np.squeeze(test_set['y']).astype('int32')-1]


    dataset.add_variable({'images':{'train_set':train_set[0],
                                    'test_set':test_set[0]},
                        'labels':{'train_set':train_set[1],
                                    'test_set':test_set[1]}})

    print('Dataset svhn loaded in','{0:.2f}'.format(time.time()-t),'s.')
    return dataset
