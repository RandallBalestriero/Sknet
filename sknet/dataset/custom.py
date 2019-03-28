import os
import pickle,gzip
import urllib.request
import numpy as np
import time



class custom(dict):
    """Grayscale digit classification.
    The `MNIST <http://yann.lecun.com/exdb/mnist/>`_ database of handwritten 
    digits, available from this page, has a training set of 60,000 examples, 
    and a test set of 10,000 examples. It is a subset of a larger set available 
    from NIST. The digits have been size-normalized and centered in a 
    fixed-size image. It is a good database for people who want to try learning
    techniques and pattern recognition methods on real-world data while 
    spending minimal efforts on preprocessing and formatting.
    """
    def __init__(self,train_set,n_classes,data_format,test_set=None,valid_set=None,**kwargs):
        """Set up the configuration for data loading and data format

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :param path:(optional, default $DATASET_PATH), the path to look for the data and 
                    where the data will be downloaded if not present
        :type path: str
        """
        dict_init = [("train_set",train_set), ("test_set",test_set),
                    ("valid_set",valid_set), ("n_classes",n_classes),
                    ("data_format",data_format)]
        super().__init__(dict_init+list(kwargs.items()))
    def load(self):
        """Load the dataset (download if necessary) and adapt
        the class attributes based on the given data format.

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :return: None, nothing is returned as the data and specific dataset informations are set as attributes
        :rtype: NoneType
        """
        pass

