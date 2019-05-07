import sys
sys.path.insert(0, "../")

import sknet
from sknet.optimize import Adam
from sknet.optimize.loss import *
from sknet.optimize import schedule
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
from sknet import ops,layers
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py
import glob

PLOT = 0
if PLOT:
    for data_augmentation in [0,1]:
        for D in [1,3,5]:
            for W in [1]:
                filename = 'cifar10_'+str(data_augmentation)+'_'+str(D)+'_'\
                                                  +str(W)+'.h5'
                if not os.path.isfile(filename):
                    continue
                f = h5py.File(filename,'r',swmr=True)

                train_loss  =f['train_set/loss/1'][...].flatten()
                train_labels=f['train_set/loss/3'][...].flatten()
                train_A     =f['train_set/loss/2'][...]
                train_A     = train_A.reshape((len(train_labels),10))

                test_loss   =f['test_set/accu/0'][...].flatten()
                test_labels =f['test_set/accu/2'][...].flatten()
                test_A      =f['test_set/accu/1'][...]
                test_A      =test_A.reshape((len(test_labels),10))

                plt.figure(figsize=(10,6))
                plt.subplot(321)
                plt.plot(train_loss)
                plt.title('Train set, every 100 batch')
                plt.xticks([])
                plt.ylabel('cross entropy')

                plt.subplot(323)
                rows = range(len(train_labels))
                plt.plot(train_A[rows,train_labels],alpha=0.3)
                plt.plot((train_A.sum(1)-train_A[rows,train_labels])/9,
                                                                     alpha=0.3)
                plt.title('Correct and Wrong class rows')
                plt.xticks([])

                plt.subplot(325)
                for k in range(10):
                    plt.semilogy(train_A[:,k],alpha=0.5)
                plt.title('Individual rows')
                plt.suptitle('Train CIFAR10, augmentation:'\
                     +str(data_augmentation)+'_depth:'+str(D)+'_width:'+str(W))

                plt.subplot(322)
                plt.plot(test_loss*100)
                plt.title('Test set, every epoch')
                plt.xticks([])
                plt.ylabel(r'accuracy in $\%$')

                plt.subplot(324)
                rows = range(len(test_labels))
                plt.semilogy(test_A[rows,test_labels], alpha=0.3)
                plt.semilogy((test_A.sum(1)-test_A[rows,test_labels])/9,
                                                                    alpha=0.3)
                plt.title('Correct and Wrong class rows')
                plt.legend(['True class','Other classes'],loc='upper left')
                plt.xticks([])

                plt.subplot(326)
                for k in range(10):
                    plt.semilogy(test_A[:,k],alpha=0.5)
                plt.title('Individual rows')
                plt.suptitle('Test CIFAR10, options:'+str(data_augmentation)\
                                                     +'_'+str(D)+'_'+str(W))
                plt.savefig('images/save_hessian'+str(data_augmentation)+'_'\
                                           +str(D)+'_'+str(W)+'_summary.pdf')
                plt.close()
                f.close()
    exit()















# Data Loading
#-------------
dataset = sknet.dataset.load_cifar10()
dataset.split_set("train_set","valid_set",0.15)

preprocess = sknet.dataset.Standardize().fit(dataset['images/train_set'])
dataset['images/train_set'] = preprocess.transform(dataset['images/train_set'])
dataset['images/test_set']  = preprocess.transform(dataset['images/test_set'])
dataset['images/valid_set'] = preprocess.transform(dataset['images/valid_set'])

dataset.create_placeholders(batch_size=32,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                        'valid_set':BatchIterator('continuous'),
                        'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape
my_layer  = layers.custom_layer(ops.Dense,ops.BatchNorm,ops.Activation)

dnn       = sknet.network.Network(name='simple_model')
data_augmentation = np.int32(sys.argv[-3])
D = np.int32(sys.argv[-2])
W = np.int32(sys.argv[-1])

if data_augmentation:
    dnn.append(ops.RandomAxisReverse(dataset.images,axis=[-1]))
    dnn.append(ops.RandomCrop(dnn[-1],(28,28),seed=10))
    start = 2
else:
    dnn.append(dataset.images)
    start = 1

noise = tf.random_normal(dnn[-1].get_shape().as_list())*0.0001

dnn.append(ops.Concat([dnn[-1],dnn[-1]+noise],axis=0))

sknet.network.Resnet(dnn,dataset.n_classes,D=D,W=W)


# Quantities
#-----------

row_A = list()
for i in range(dataset.n_classes):
    grad = tf.gradients(dnn[-1],dnn[start], tf.ones((64,1))*tf.one_hot(i,10))
    row_A.append(tf.reduce_sum(tf.square(grad[0][:32]-grad[0][32:]),[1,2,3]))

prediction = dnn[-1]
loss       = crossentropy_logits(p=dataset.labels,q=prediction[:32])
hessian    = tf.stack(row_A,1) #(32,n_classes)
accu       = accuracy(dataset.labels,prediction[:32])

B         = dataset.N('train_set')//32
lr        = sknet.optimize.PiecewiseConstant(0.005,
                                    {100*B:0.003,200*B:0.001,250*B:0.0005})
optimizer = Adam(loss,lr,params=dnn.variables(trainable=True))
minimizer = tf.group(optimizer.updates+dnn.updates)


# Workers
#---------

minimize = sknet.Worker(name='loss',context='train_set',
            op=[minimizer,loss,hessian,dataset.labels],
            deterministic=False,period=[1,100,100,100])

accu     = sknet.Worker(name='accu',context='test_set',
            op=[accu,hessian,dataset.labels],
            deterministic=True, transform_function=[np.mean,None,None])

queue = sknet.Queue((minimize, accu),filename='cifar10_'\
                    +str(data_augmentation)+'_'+str(D)+'_'+str(W)+'.h5')

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_queue(queue,repeat=350)




