import os
from zipfile import ZipFile
import urllib.request
import numpy as np
import scipy.misc
from PIL import Image
from . import Dataset


def load_tinyimagenet(data_format='NCHW'):

    PATH = os.environ['DATASET_PATH']

    if not os.path.isdir(PATH+'tinyimagenet'):
        os.mkdir(PATH+'tinyimagenet')

    if not os.path.exists(PATH+'tinyimagenet/tiny-imagenet-200.zip'):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        urllib.request.urlretrieve(url,PATH+'tinyimagenet/tiny-imagenet-200.zip')
    # Loading the file
    f       = ZipFile(PATH+'tinyimagenet/tiny-imagenet-200.zip', 'r')
    names   = [name for name in f.namelist() if name.endswith('JPEG')]
    val_classes = np.loadtxt(f.open('tiny-imagenet-200/val/val_annotations.txt'),
                             dtype=str, delimiter='\t')
    val_classes = dict([(a, b) for a, b in zip(val_classes[:, 0], val_classes[:, 1])])
    x_train, x_test, x_valid, y_train, y_test, y_valid = [], [], [], [], [], []
    for name in names:
        if 'train' in name:
            classe = name.split('/')[-1].split('_')[0]
            x_train.append(scipy.misc.imread(f.open(name), flatten=False,
                           mode='RGB').transpose((2, 0, 1)))
            y_train.append(classe)
        if 'val' in name:
            x_valid.append(scipy.misc.imread(f.open(name), flatten=False,
                           mode='RGB').transpose((2, 0, 1)))
            arg =  name.split('/')[-1]
            y_valid.append(val_classes[arg])
        if 'test' in name:
            x_test.append(scipy.misc.imread(f.open(name), flatten=False,
                           mode='RGB').transpose((2, 0, 1)))

    # as per loading, the classes are of the form 'n023552' and thus need to be
    # converted to an integer. The mapping dictionnary will do the one to one
    # correspondance.
    mapping = dict()
    counter = 0
    for classes in np.unique(y_train):
        if classes not in mapping:
            mapping[classes] = counter
            counter += 1

    y_train = np.asarray([mapping[y] for y in y_train]).astype('int32')
    y_valid = np.asarray([mapping[y] for y in y_valid]).astype('int32')
    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')
    x_valid = np.asarray(x_valid).astype('float32')

    dataset = Dataset()
    dataset['images/train_set'], dataset['labels/train_set'] = x_train, y_train
    dataset['images/test_set'], dataset['labels/test_set'] = x_test, np.zeros(100, dtype='int32')
    dataset['images/valid_set'], dataset['labels/valid_set'] = x_valid, y_valid

    dataset.n_classes = 200

    return dataset






