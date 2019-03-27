import os
import pickle,gzip
import urllib.request
import numpy as np
import tarfile
import time





class cifar10:
    """
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
    def __init__(self,data_format = 'NCHW'):
        self.data_format     = data_format
        self.given_test_set  = True
        self.given_train_set = True
        self.given_valid_set = False
        self.given_unlabeled = False
        self.name            = 'cifar10'
        self.n_classes       = 10
        self.classes = {0:"airplane",
                        1:"automobile",
                        2:"bird",
                        3:"cat",
                        4:"deer",
                        5:"dog",
                        6:"frog",
                        7:"horse",
                        8:"ship",
                        9:"truck"}
    def load(self):

        t = time.time()

        PATH = os.environ['DATASET_PATH']

        # Check if directory exists
        if not os.path.isdir(PATH+'cifar10'):
            os.mkdir(PATH+'cifar10')
            print('Creating Directory')

        # Check if file exists
        if not os.path.exists(PATH+'cifar10/cifar10.tar.gz'):
            print('Downloading Files')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            urllib.request.urlretrieve(url,PATH+'cifar10/cifar10.tar.gz')

        # Loading dataset
        print('Loading CIFAR10')
        tar = tarfile.open(PATH+'cifar10/cifar10.tar.gz', 'r:gz')
        # Loop over ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5',]
        train_names = ['cifar-10-batches-py/data_batch_1',
                'cifar-10-batches-py/data_batch_2',
                'cifar-10-batches-py/data_batch_3',
                'cifar-10-batches-py/data_batch_4',
                'cifar-10-batches-py/data_batch_5']
        test_name  = ['cifar-10-batches-py/test_batch']
        train_set  = [[],[]]
        test_set   = [[],[]]
        for member in tar.getmembers():
            if member.name in train_names:
                f       = tar.extractfile(member)
                content = f.read()
                data_dic= pickle.loads(content,encoding='latin1')
                train_set[0].append(data_dic['data'].reshape((-1,3,32,32)))
                train_set[1].append(data_dic['labels'])
            elif member.name in test_name:
                f       = tar.extractfile(member)
                content = f.read()
                data_dic= pickle.loads(content,encoding='latin1')
                test_set= [data_dic['data'].reshape((-1,3,32,32)),
                            data_dic['labels']]
        train_set[0] = np.concatenate(train_set[0],axis=0)
        train_set[1] = np.concatenate(train_set[1],axis=0)
        # Check formatting
        if data_format=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
            self.datum_shape  = (32,32,3)
        else:
            self.datum_shape  = (3,32,32)
        
        self.train_set = train_set
        self.test_set  = test_set

        print('Dataset CIFAR10 loaded in','{0:.2f}'.format(time.time()-t),'s.')




