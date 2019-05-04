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

dataset = sknet.dataset.load_freefield1010(subsample=2)
dataset['signals']['train_set']-=dataset['signals']['train_set'].mean(2,keepdims=True)
dataset['signals']['train_set']/=dataset['signals']['train_set'].max(2,keepdims=True)

dataset.split_set("train_set","test_set",0.33)
#dataset.preprocess(sknet.dataset.Standardize,data="images",axis=[0])


dataset.create_placeholders(batch_size=16,
       iterators_dict={'train_set':BatchIterator("random_see_all"),
                       'valid_set':BatchIterator('continuous'),
                       'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape

my_layer = layers.custom_layer(ops.Dense,ops.BatchNorm,ops.Activation)

dnn = sknet.network.Network(name='model_base')

dnn.append(ops.SplineWaveletTransform(dataset.signals,J=5,Q=16,K=15))
dnn.append(ops.Pool2D(tf.log(dnn[-1]+0.001),(1,1024),strides=(1,512),pool_type='AVG'))
dnn.append(ops.BatchNorm(dnn[-1],[0,3]))


dnn.append(layers.Conv2DPool(dnn[-1],[(32,3,5),{'b':None}],
                                [[0,2,3]],[0.1],[(1,3)]))

dnn.append(layers.Conv2DPool(dnn[-1],[(64,5,1),{'b':None}],
                                [[0,2,3]],[0.1],[(3,1)]))

dnn.append(layers.Conv2D(dnn[-1],[(64,1,1),{'b':None}],
                                [[0,2,3]],[0.1]))

dnn.append(layers.Conv2DPool(dnn[-1],[(64,3,5),{'b':None}],
                                  [[0,2,3]],[0.1],[(1,3)]))

dnn.append(layers.Conv2DPool(dnn[-1],[(64,5,1),{'b':None}],
                                  [[0,2,3]],[0.1],[(3,1)]))

dnn.append(layers.Conv2D(dnn[-1],[(128,1,1),{'b':None}],
                                  [[0,2,3]],[0.1]))

dnn.append(layers.Conv2DPool(dnn[-1],[(128,1,3),{'b':None}],
                                  [[0,2,3]],[0.1],[(1,3)]))

dnn.append(my_layer(dnn[-1],[256,{'b':None}],
                                    [[0]],[0]))

dnn.append(my_layer(dnn[-1],[128,{'b':None}],
                                    [[0]],[0]))

dnn.append(ops.Dense(dnn[-1],units=dataset.n_classes))


# Loss and Optimizer
#-------------------

# Compute some quantities that we want to keep track and/or act upon
loss     = crossentropy_logits(p=dataset.labels,q=dnn[-1])
accuracy = accuracy(labels=dataset.labels,predictions=dnn[-1])
auc      = AUC(dataset.labels,tf.nn.softmax(dnn[-1])[:,1])
#print(auc)
#input('try')

# we aim at minimizing the loss, so we create the optimizer (Adam in this case)
# with a stepwise learning rate, the minimizer is the operation that applies
# the changes to the model parameters, we also specify that this process
# should also include some possible network dependencies present in UPDATE_OPS

optimizer = sknet.optimize.Adam(loss,0.00501,params=dnn.params)
minimizer = tf.group(optimizer.updates+dnn.updates)

# Workers
#---------

work1 = sknet.Worker(name='minimizer',context='train_set',op=[minimizer,loss],
        deterministic=False)

work2 = sknet.Worker(name='AUC',context='test_set',op=[accuracy,auc],
        deterministic=True, transform_function=np.mean,verbose=2)

queue = sknet.Queue((work1,work2),filename='test_bird.h5')

# Pipeline
#---------

# the pipeline is assembling all the components for executing the program,
# the dataset, the workers and the linkage representing what missing values
# from the network have to be searched for in the dataset 
# (for example, the labels)

workplace = sknet.utils.Workplace(dnn,dataset=dataset)

# will fit the model for 50 epochs and return the gathered op
# outputs given the above definitions

output = workplace.execute_queue(queue,repeat=200)


