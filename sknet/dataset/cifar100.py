import urllib.request
import numpy as np
import tarfile
import os
import pickle
import time

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


class cifar100(dict):
    """Image classification.
    The `CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset is 
    just like the CIFAR-10, except it has 100 classes containing 600 images 
    each. There are 500 training images and 100 testing images per class. 
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each 
    image comes with a "fine" label (the class to which it belongs) and a 
    "coarse" label (the superclass to which it belongs).
    """
    def __init__(self,data_format='NCHW',path=None,default_classes='fine_labels'):
        """Set up the configuration for data loading and data format

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :param path:(optional, default $DATASET_PATH), the path to look for the data and 
                    where the data will be downloaded if not present
        :type path: str
        """
        if path is None:
            path = os.environ['DATASET_PATH']
        if data_format=='NCHW':
            datum_shape = (3,32,32)
        else:
            datum_shape = (32,32,3)
        if default_classes=='fine_labels':
            classes = 100
        else:
            classes = 20
        dict_init = [("train_set",None),("test_set",None),
                    ("datum_shape",datum_shape),("n_classes",classes),
                    ("n_channels",3),("spatial_shape",(32,32)),
                    ("path",path),("data_format",data_format),("name","cifar100"),
                    ("classes",labels_list)]
        super().__init__(dict_init)
 
    def load(self):
        """Load the dataset (download if necessary) and adapt
        the class attributes based on the given data format.

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :return: return the train and test set, each as a couple (images,labels)
        :rtype: [(train_images,train_coarse_labels,train_fine_labels),
                (test_images,test_coarse_labels,test_fine_labels)]
        """
        print('Loading cifar100')
                
        t = time.time()

        PATH = self["path"]

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
        print('\tOpening cifar100')
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
        # Check formating
        if self["data_format"]=='NHWC':
            train_set[0]     = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]      = np.transpose(test_set[0],[0,2,3,1])

        self["train_set"] = train_set
        self["test_set"]  = test_set

        print('Dataset cifar100 loaded in','{0:.2f}'.format(time.time()-t),'s.')
