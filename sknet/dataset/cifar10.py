import os
import pickle,gzip
import urllib.request
import numpy as np
import tarfile
import time
from . import Dataset


class cifar10(Dataset):
    """Image classification.
    The `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset 
    was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey 
    Hinton. It consists of 60000 32x32 colour images in 10 classes, with 
    6000 images per class. There are 50000 training images and 10000 test images. 
    The dataset is divided into five training batches and one test batch, 
    each with 10000 images. The test batch contains exactly 1000 randomly
    selected images from each class. The training batches contain the 
    remaining images in random order, but some training batches may 
    contain more images from one class than another. Between them, the 
    training batches contain exactly 5000 images from each class. 
    """
    def __init__(self,data_format='NCHW',path=None):
        """Set up the configuration for data loading and data format

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :param path: (optional, default :envvar:`$DATASET_PATH`), the path to look for the data and 
                     where the data will be downloaded if not present
        :type path: str
        """
        if path is None:
            path = os.environ['DATASET_PATH']
        if data_format=='NCHW':
            datum_shape = (3,32,32)
        else:
            datum_shape = (32,32,3)
        dict_init = [("train_set",None),("test_set",None),
                    ("datum_shape",datum_shape),("n_classes",10),
                    ("n_channels",3),("spatial_shape",(32,32)),
                    ("path",path),("data_format",data_format),("name","cifar10")]
        class_init = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                        "frog", "horse", "ship", "truck"]
        super().__init__(dict_init+[('classes',class_init)])

    def load(self):
        """Load the dataset (download if necessary) and set
        the class attributes.
        """
        t = time.time()

        print('Loading cifar10')

        PATH = self["path"]

        # Check if directory exists
        if not os.path.isdir(PATH+'cifar10'):
            print('\tCreating Directory')
            os.mkdir(PATH+'cifar10')

        # Check if file exists
        if not os.path.exists(PATH+'cifar10/cifar10.tar.gz'):
            print('\tDownloading cifar10 Dataset...')
            td = time.time()
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            urllib.request.urlretrieve(url,PATH+'cifar10/cifar10.tar.gz')
            print("\tDone in {:.2f} s.".format(time.time()-td))

        # Loading dataset
        print('\tOpening cifar10')
        tar = tarfile.open(PATH+'cifar10/cifar10.tar.gz', 'r:gz')

        # Load training set
        train_images  = list()
        train_labels  = list()
        train_names = ['cifar-10-batches-py/data_batch_1',
                'cifar-10-batches-py/data_batch_2',
                'cifar-10-batches-py/data_batch_3',
                'cifar-10-batches-py/data_batch_4',
                'cifar-10-batches-py/data_batch_5']
        for names in train_names:
            f        = tar.extractfile('cifar-10-batches-py/data_batch_1').read()
            data_dic = pickle.loads(f,encoding='latin1')
            train_images.append(data_dic['data'].reshape((-1,3,32,32)))
            train_labels.append(data_dic['labels'])
        train_set = [np.concatenate(train_images,0),
                        np.concatenate(train_labels,0)]

        # Load testing set
        test_images = list()
        test_labels = list()
        f        = tar.extractfile('cifar-10-batches-py/test_batch').read()
        data_dic = pickle.loads(f,encoding='latin1')
        test_set = [data_dic['data'].reshape((-1,3,32,32)),
                            data_dic['labels']]

        # Check formatting
        if self["data_format"]=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])

        self["train_set"]= train_set
        self["test_set"] = test_set

        print('Dataset cifar10 loaded in','{0:.2f}'.format(time.time()-t),'s.')
