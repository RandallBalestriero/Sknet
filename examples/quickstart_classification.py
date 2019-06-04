import sys
sys.path.insert(0, "../")

import sknet
import matplotlib
matplotlib.use('Agg')
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pylab as pl
import time
import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops, layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="the dataset to train on, can be"\
           +"mnist, fasionmnist, cifar10, ...", type=str, default='mnist')
parser.add_argument('--model', help="the model to use: cnn or resnet",
                      type=str, choices=['cnn', 'resnet'], default='cnn')
parser.add_argument("--data_augmentation", help="using data augmentation",
                     type=sknet.utils.str2bool, default='0')

args = parser.parse_args()
DATASET = args.dataset
MODEL = args.model
DATA_AUGMENTATION = args.data_augmentation

# Data Loading
if DATASET=='mnist':
    dataset = sknet.dataset.load_mnist()
elif DATASET=='fashionmnist':
    dataset = sknet.dataset.load_fashonmnist()
elif DATASET=='cifar10':
    dataset = sknet.dataset.load_cifar10()
elif DATASET=='cifar100':
    dataset = sknet.dataset.load_cifar100()
elif DATASET=='svhn':
    dataset = sknet.dataset.load_svhn()

if "valid_set" not in dataset.sets:
    dataset.split_set("train_set", "valid_set", 0.15)

standardize = sknet.dataset.Standardize().fit(dataset['images/train_set'])
dataset['images/train_set'] = \
                        standardize.transform(dataset['images/train_set'])
dataset['images/test_set'] = \
                        standardize.transform(dataset['images/test_set'])
dataset['images/valid_set'] = \
                        standardize.transform(dataset['images/valid_set'])

iterator = BatchIterator(32, {'train_set': 'random_see_all',
                              'valid_set': 'continuous',
                              'test_set': 'continuous'})

dataset.create_placeholders(iterator, device="/cpu:0")

# Create Network

dnn = sknet.Network()

if DATA_AUGMENTATION:
    dnn.append(ops.RandomAxisReverse(dataset.images, axis=[-1]))
    dnn.append(ops.RandomCrop(dnn[-1], (28, 28), seed=10))
else:
    dnn.append(dataset.images)

if MODEL == 'cnn':
    sknet.networks.ConvLarge(dnn, dataset.n_classes)
elif MODEL == 'resnet':
    sknet.networks.Resnet(dnn, dataset.n_classes, D=4, W=1)

prediction = dnn[-1]

loss = sknet.losses.crossentropy_logits(dataset.labels, prediction)
accu = sknet.losses.accuracy(dataset.labels, prediction)

B = dataset.N('train_set')//32
lr = sknet.schedules.PiecewiseConstant(0.05, {100*B: 0.005, 200*B: 0.001,
                                              250*B: 0.0005})

optimizer = sknet.optimizers.NesterovMomentum(loss, lr,
                                    params=dnn.variables(trainable=True))
minimizer = tf.group(optimizer.updates+dnn.updates)

# Workers
#---------

def period_function(batch, epoch):
    if batch%100 == 0:
        return True
    return False

train = sknet.Worker(context='train_set', deterministic=False,
                     minimizer=minimizer, loss=(loss, period_function))

test = sknet.Worker(context='test_set', deterministic=True, accuracy=accu)

queue = sknet.Queue((train, test))

workplace = sknet.utils.Workplace(dataset)
workplace.execute_queue(queue, repeat=350, deter_func=dnn.deter_dict)


