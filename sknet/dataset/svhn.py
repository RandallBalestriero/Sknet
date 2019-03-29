import scipy.io as sio
import os
import pickle,gzip
import urllib.request
import numpy as np
import time



class svhn(dict):
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
            datum_shape = (3,32,32)
        else:
            datum_shape = (32,32,3)
        dict_init = [("train_set",None),("test_set",None),
                    ("datum_shape",datum_shape),("n_classes",10),
                    ("n_channels",3),("spatial_shape",(32,32)),
                    ("path",path),("data_format",data_format),("name","svhn"),
                    ("classes",[str(1+u) for u in range(10)])]
        super().__init__(dict_init)

    def load(self):
        """Load the dataset (download if necessary) and set
        the class attributess.
        """
        print('Loading svhn')

        t = time.time()

        PATH = self["path"]

        if not os.path.isdir(PATH+'svhn'):
            os.mkdir(PATH+'svhn')
            print('\tCreating svhn Directory')

        if not os.path.exists(PATH+'svhn/train_32x32.mat'):
            print('\tDownloading svhn Train Set')
            td = time.time()
            url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
            urllib.request.urlretrieve(url,PATH+'svhn/train_32x32.mat')
            print("\tDone in {:.2f} s.".format(time.time()-td))

        if not os.path.exists(PATH+'svhn/test_32x32.mat'):
            print('\tDownloading svhn Test Set')
            td = time.time()
            url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
            urllib.request.urlretrieve(url,PATH+'svhn/test_32x32.mat')
            print("\tDone in {:.2f} s.".format(time.time()-td))

        print('\tOpening svhn')
        train_set = sio.loadmat(PATH+'svhn/train_32x32.mat')
        train_set = [train_set['X'].transpose([3,2,0,1]).astype('float32'),
                    np.squeeze(train_set['y']).astype('int32')-1]
        test_set  = sio.loadmat(PATH+'svhn/test_32x32.mat')
        test_set  = [test_set['X'].transpose([3,2,0,1]).astype('float32'),
                    np.squeeze(test_set['y']).astype('int32')-1]

        # Check formatting
        if self["data_format"]=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])

        self["train_set"]= train_set
        self["test_set"] = test_set

        print('Dataset svhn loaded in','{0:.2f}'.format(time.time()-t),'s.')
