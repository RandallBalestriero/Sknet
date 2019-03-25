import os
import pickle,gzip
import urllib.request
import numpy as np
import tarfile
import time





class cifar10:
    def __init__(self):
        pass
    def load(self,data_format='NCHW', seed=None):

        self.name        = 'cifar10'
        self.classes     = 10
        self.data_format = data_format

        t = time.time()

        PATH = os.environ['DATASET_PATH']

        if not os.path.isdir(PATH+'cifar10'):
            os.mkdir(PATH+'cifar10')
            print('Creating Directory')

        if not os.path.exists(PATH+'cifar10/cifar10.tar.gz'):
            print('Downloading Files')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            urllib.request.urlretrieve(url,PATH+'cifar10/cifar10.tar.gz')

        print('Loading CIFAR10')
        # Loading the file
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
        # Compute a Valid Set
        random_indices = np.random.RandomState(seed=seed).permutation(train_set[0].shape[0])
        train_indices  = random_indices[:int(train_set[0].shape[0]*0.9)]
        valid_indices  = random_indices[int(train_set[0].shape[0]*0.9):]
        valid_set      = [train_set[0][valid_indices],train_set[1][valid_indices]]
        train_set      = [train_set[0][train_indices],train_set[1][train_indices]]
        # Check formatting
        if data_format=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
            valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
            image_shape  = (32,32,3)
        else:
            image_shape  = (3,32,32)

        print('Dataset CIFAR10 loaded in','{0:.2f}'.format(time.time()-t),'s.')
        return train_set,valid_set,test_set




