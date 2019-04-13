import urllib.request
import numpy as np
import tarfile
import os
import pickle
import time

from . import Dataset
from ..utils import to_one_hot

from . import Dataset

labels_list = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


def load_cifar100(PATH=None):
    """Image classification.
    The `CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset is 
    just like the CIFAR-10, except it has 100 classes containing 600 images 
    each. There are 500 training images and 100 testing images per class. 
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each 
    image comes with a "fine" label (the class to which it belongs) and a 
    "coarse" label (the superclass to which it belongs).

    :param path: (optional, default $DATASET_PATH), the path to look for the data and 
                 where the data will be downloaded if not present
    :type path: str
    """

    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    dict_init = [("n_classes",100),("path",PATH),("name","cifar100"),
                ("classes",labels_list),("n_coarse_classes",20)]
    dataset = Dataset(**dict(dict_init))
    
    # Load the dataset (download if necessary) and set
    # the class attributes.
        
    print('Loading cifar100')
                
    t = time.time()

    if not os.path.isdir(PATH+'cifar100'):
        print('\tCreating cifar100 Directory')
        os.mkdir(PATH+'cifar100')

    if not os.path.exists(PATH+'cifar100/cifar100.tar.gz'):
        print('\tDownloading cifar100 Dataset...')
        td = time.time()
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        urllib.request.urlretrieve(url,PATH+'cifar100/cifar100.tar.gz')
        print("\tDone in {:.2f}".format(time.time()-td))

    # Loading the file
    tar = tarfile.open(PATH+'cifar100/cifar100.tar.gz', 'r:gz')

    # Loading training set
    f         = tar.extractfile('cifar-100-python/train').read()
    data_dic  = pickle.loads(f,encoding='latin1')

    train_set = [data_dic['data'].reshape((-1,3,32,32)),
                    np.array(data_dic['coarse_labels']),
                    np.array(data_dic['fine_labels'])]

    # Loading test set
    f        = tar.extractfile('cifar-100-python/test').read()
    data_dic = pickle.loads(f,encoding='latin1')
    test_set = [data_dic['data'].reshape((-1,3,32,32)),
                    np.array(data_dic['coarse_labels']),
                    np.array(data_dic['fine_labels'])]

    dataset.add_variable({'images':[{'train_set':train_set[0],
                                    'test_set':test_set[0]},
                                    (3,32,32),'float32'],
                        'labels':[{'train_set':train_set[2],
                                    'test_set':test_set[2]},
                                    (),'int32'],
                        'coarse_labels':[{'train_set':train_set[1],
                                        'test_set':test_set[1]},
                                        (),'int32']})

    print('Dataset cifar100 loaded in','{0:.2f}'.format(time.time()-t),'s.')
    return dataset
