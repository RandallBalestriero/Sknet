import os
import gzip
import urllib.request
import numpy as np
import time





class fashionmnist(dict):
    """Grayscale `Zalando <https://jobs.zalando.com/tech/>`_ 's article image classification.
    `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ is 
    a dataset of `Zalando <https://jobs.zalando.com/tech/>`_ 's article 
    images consisting of a training set of 60,000 examples and a test set 
    of 10,000 examples. Each example is a 28x28 grayscale image, associated 
    with a label from 10 classes. We intend Fashion-MNIST to serve as a direct
    drop-in replacement for the original MNIST dataset for benchmarking 
    machine learning algorithms. It shares the same image size and structure 
    of training and testing splits.
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
        dict_init = [("train_set",None),("test_set",None),
                    ("datum_shape",datum_shape),("n_classes",10),
                    ("n_channels",1),("spatial_shape",(28,28)),
                    ("path",path),("data_format",data_format),("name","fashionmnist")]
        classes   = ["T-shirt/top", "Trouser", "Pullover","Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        super().__init__(dict_init+[('classes',classes)])
    def load(self):
        """Load the dataset (download if necessary) and adapt
        the class attributes based on the given data format.

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :return: return the train and test set, each as a couple (images,labels)
        :rtype: [(train_images,train_labels),(test_images,test_labels)]
        """
        print('Loading fashionmnist')

        t = time.time()

        PATH = self["path"]

        if not os.path.isdir(PATH+'fashionmnist'):
            print('\tCreating Directory')
            os.mkdir(PATH+'fashionmnist')

        if not os.path.exists(PATH+'fashionmnist/train-images.gz'):
            print('\tDownloading fashionmnist Train Images...')
            td  = time.time() 
            url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
            urllib.request.urlretrieve(url,PATH+'fashionmnist/train-images.gz')
            print("\tDone in {:.2f} s.".format(time.time()-td))

        if not os.path.exists(PATH+'fashionmnist/train-labels.gz'):
            print('\tDownloading fashionmnist Train Labels...')
            td = time.time()
            url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
            urllib.request.urlretrieve(url,PATH+'fashionmnist/train-labels.gz')
            print("\tDone in {:.2f} s.".format(time.time()-td))

        if not os.path.exists(PATH+'fashionmnist/test-images.gz'):
            print('\tDownloading fashionmnist Test Images...')
            td = time.time()
            url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'
            urllib.request.urlretrieve(url,PATH+'fashionmnist/test-images.gz')
            print("\tDone in {:.2f} s.".format(time.time()-td))

        if not os.path.exists(PATH+'fashionmnist/test-labels.gz'):
            print('\tDownloading fashionmnist Test Labels...')
            td = time.time()
            url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
            urllib.request.urlretrieve(url,PATH+'fashionmnist/test-labels.gz')
            print("\tDone in {:.2f} s.".format(time.time()-td))

        # Loading the file
        print('\tOpening fashionmnist')

        with gzip.open(PATH+'fashionmnist/train-labels.gz', 'rb') as lbpath:
            train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(PATH+'fashionmnist/train-images.gz', 'rb') as lbpath:
            train_images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16).reshape((-1,1,28,28))

        with gzip.open(PATH+'fashionmnist/test-labels.gz', 'rb') as lbpath:
            test_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(PATH+'fashionmnist/test-images.gz', 'rb') as lbpath:
            test_images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16).reshape((-1,1,28,28))

        # Check Formatting
        if self["data_format"]=='NHWC':
            train_images = np.transpose(train_images,[0,2,3,1])
            test_images  = np.transpose(test_images,[0,2,3,1])

        self["train_set"] = [train_images,train_labels]
        self["test_set"]  = [test_images,test_labels]

        print('Dataset fashionmnist loaded in','{0:.2f}'.format(time.time()-t),'s.')
