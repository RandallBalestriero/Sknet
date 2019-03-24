import os
from zipfile import ZipFile
import urllib.request
import numpy as np
import scipy.misc



def load(data_format='NCHW'):

    PATH = os.environ['DATASET_PATH']

    if not os.path.isdir(PATH+'tinyimagenet'):
        os.mkdir(PATH+'tinyimagenet')

    if not os.path.exists(PATH+'tinyimagenet/tiny-imagenet-200.zip'):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        urllib.request.urlretrieve(url,PATH+'tinyimagenet/tiny-imagenet-200.zip')
    # Loading the file
    f       = ZipFile(PATH+'tinyimagenet/tiny-imagenet-200.zip', 'rb')
    names   = f.namelist()
    val_classes = f.read('tiny-imagenet-200/val/val_annotations.txt')
    for name in names:
        if 'train' in name:
            classe = name.split('/')[-1].split('_')[0]
            x_train.append(scipy.misc.imread(f.read(name), flatten=False, mode='RGB'))
            y_train.append(classe)
        if 'val' in name:
            x_valid.append(scipy.misc.imread(f.read(name), flatten=False, mode='RGB'))
            y_valid.append(val_classes[name])
        if 'test' in name:
            x_test.append(scipy.misc.imread(f.read(name), flatten=False, mode='RGB'))


    # Check formatting
    if data_format=='NHWC':
        train_set[0] = np.transpose(train_set[0],[0,2,3,1])
        test_set[0]  = np.transpose(test_set[0],[0,2,3,1])
        valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
    return train_set, valid_set, test_set






