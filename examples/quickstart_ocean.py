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


# Data Loading
#-------------

dataset = sknet.dataset.load_DCLDE(window_size=1024*100)
dataset['signals']['train_set']/=dataset['signals']['train_set'].max()

for signal in dataset['signals']['train_set']:
    plot(signal)
    show()


print(dataset['signals']['train_set'])
dataset.split_set("train_set","test_set",0.20)
#dataset.preprocess(sknet.dataset.Standardize,data="images",axis=[0])


dataset.create_placeholders(batch_size=16,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                        'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape

my_layer = layers.custom_layer(ops.Dense,ops.BatchNorm,ops.Activation)

dnn = sknet.network.Network(name='model_base')

dnn.append(ops.SplineWaveletTransform(dataset.signals,J=3,Q=16,K=11))
dnn.append(ops.Inverse(dnn[-1],dnn[-1]))

# Loss and Optimizer
#-------------------

# Compute some quantities that we want to keep track and/or act upon
loss     = MSE(dataset.signals,dnn[-1])

optimizer = sknet.optimize.Adam(loss,0.0001,params=dnn.params)
minimizer = tf.group(optimizer.updates+dnn.updates)

# Workers
#---------

work1 = sknet.Worker(op_name='minimizer',context='train_set',op=minimizer, 
        instruction='execute every batch', deterministic=False)

work2 = sknet.Worker(op_name='loss',context='train_set',op=loss,
        instruction='save & print every 100 batch', deterministic=False,
        description='saving the loss every 100 batches')

work3 = work2.alter(context='test_set',deterministic=True)


queue = sknet.Queue((work1+work2,work3))

# Pipeline
#---------

workplace = sknet.utils.Workplace(dnn,dataset=dataset)

output = workplace.execute_queue(queue,repeat=20,
                            filename='test.h5',save_period=20)


