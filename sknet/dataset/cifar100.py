import urllib.request
import numpy as np
import tarfile
import os
import pickle
import time




class cifar100:
    def __init__(self):
        pass
    def load(self,data_format='NCHW',fine_labels=True, seed=None):
                
        self.name          = 'cifar100'
        self.data_format   = data_format

        if fine_labels:
            self.classes = 100
        else:
            self.classes = 10

        t = time.time()

        label = 'fine_labels' if fine_labels else 'coarse_labels'

        PATH = os.environ['DATASET_PATH']

        if not os.path.isdir(PATH+'cifar100'):
            os.mkdir(PATH+'cifar100')
            print('Creating Directory')
        if not os.path.exists(PATH+'cifar100/cifar100.tar.gz'):
            print('Downloading CIFAR100 Files')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            urllib.request.urlretrieve(url,PATH+'cifar100/cifar100.tar.gz')
        # Loading the file
        print('Loading CIFAR100')
        tar = tarfile.open(PATH+'cifar100/cifar100.tar.gz', 'r:gz')
        train_names = ['cifar-100-python/train']
        test_name   = ['cifar-100-python/test']
        for member in tar.getmembers():
            if member.name in train_names:
                f       = tar.extractfile(member)
                content = f.read()
                data_dic= pickle.loads(content,encoding='latin1')
                train_set=[data_dic['data'].reshape((-1,3,32,32)),
                            np.array(data_dic[label])]
            elif member.name in test_name:
                f       = tar.extractfile(member)
                content = f.read()
                data_dic= pickle.loads(content,encoding='latin1')
                test_set= [data_dic['data'].reshape((-1,3,32,32)),
                            np.array(data_dic[label])]
        # Compute a Valid Set
        random_indices = np.random.RandomState(seed=seed).permutation(train_set[0].shape[0])
        train_indices  = random_indices[:int(train_set[0].shape[0]*0.9)]
        valid_indices  = random_indices[int(train_set[0].shape[0]*0.9):]
        valid_set    = [train_set[0][valid_indices],train_set[1][valid_indices]]
        train_set    = [train_set[0][train_indices],train_set[1][train_indices]]
        # Check formating
        if data_format=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0] = np.transpose(test_set[0],[0,2,3,1])
            valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
            self.image_shape = (32,32,3)
        else:
            self.image_shape = (3,32,32)
        print('Dataset CIFAR100 loaded in','{0:.2f}'.format(time.time()-t),'s.')
        return train_set,valid_set,test_set


