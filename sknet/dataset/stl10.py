import urllib.request
import time
import sys
import os, sys, tarfile, io
import numpy as np
import matplotlib.pyplot as plt
    

class stl10(dict):
    """ Image classification with extra unlabeled images.
    The `STL-10 <https://cs.stanford.edu/~acoates/stl10/>`_ dataset is an image 
    recognition dataset for developing unsupervised feature learning, 
    deep learning, self-taught learning algorithms. It is inspired by the 
    CIFAR-10 dataset but with 
    some modifications. In particular, each class has fewer labeled 
    training examples than in CIFAR-10, but a very 
    large set of unlabeled examples is provided to learn image models prior 
    to supervised training. The primary challenge is to make use of the 
    unlabeled data (which comes from a similar but different distribution from 
    the labeled data) to build a useful prior. We also expect that the higher 
    resolution of this dataset (96x96) will make it a challenging benchmark 
    for developing more scalable unsupervised learning methods.
    """
    def __init__(self,data_format='NCHW',path=None):
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
        dict_init = [("train_set",None),("test_set",None),
                    ("datum_shape",datum_shape),("n_classes",10),
                    ("n_channels",3),("spatial_shape",(32,32)),
                    ("path",path),("data_format",data_format),("name","stl10")]
        classes      = ["airplane", "bird", "car", "cat", "deer", "dog",
                        "horse", "monkey", "ship", "truck"]
        super().__init__(dict_init+[('classes',classes)])

    def load(self):
        """Load the dataset (download if necessary) and adapt
        the class attributes based on the given data format.

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :return: return the train and test set, each as a couple (images,labels) 
                 and the unlabeled images 
        :rtype: [(train_images,train_labels),
                (test_images,test_labels),unlabeled_images]
        """
        print("Loading stl10")
        t    = time.time()
        PATH = self["path"]


        # Check if directory exists
        if not os.path.isdir(PATH+'stl10'):
            print('\tCreating stl10 Directory')
            os.mkdir(PATH+'stl10')

        # Check if data file exists
        if not os.path.exists(PATH+'stl10/stl10_binary.tar.gz'):
            td = time.time()
            print('\tDownloading stl10 Dataset...')
            url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
            urllib.request.urlretrieve(url,PATH+'stl10/stl10_binary.tar.gz')  
            print('\tDone in {:.2f} s.'.format(time.time()-td))

        # Loading Dataset
        print('\tOpening stl10')
        file_ = tarfile.open(PATH+'stl10/stl10_binary.tar.gz', 'r:gz')
        # loading test label
        read_file = file_.extractfile('stl10_binary/test_y.bin').read()
        test_y = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8)-1
        # loading train label
        read_file = file_.extractfile('stl10_binary/train_y.bin').read()
        train_y = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8)-1
        # load test images
        read_file = file_.extractfile('stl10_binary/test_X.bin').read()
        test_X = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8).reshape((-1,3,96,96))
        # load train images
        read_file = file_.extractfile('stl10_binary/train_X.bin').read()
        train_X = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8).reshape((-1,3,96,96))
        # load unlabelled images
        read_file = file_.extractfile('stl10_binary/unlabeled_X.bin').read()
        unlabeled_X = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8).reshape((-1,3,96,96))

        # Check formatting
        if self["data_format"]=='NHWC':
            train_X     = np.transpose(train_X,[0,2,3,1])
            test_X      = np.transpose(test_X,[0,2,3,1])
            unlabeled_X = np.transpose(unlabeled_X,[0,2,3,1])
            
        self["train_set"] = [train_X,train_y]
        self["test_set"]  = [test_X,test_y]
        self["unlabeled"] = unlabeled_X
        print('Dataset stl10 loaded in','{0:.2f}'.format(time.time()-t),'s.')



