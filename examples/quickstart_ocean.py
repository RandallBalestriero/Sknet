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

dataset = sknet.dataset.load_DCLDE(subsample=4)

dataset.split_set("train_set","valid_set",0.33)
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

dnn.append(ops.SplineWaveletTransform(dataset.signals,J=3,Q=16,K=11))
dnn.append(ops.Pool(dnn[-1],(1,1,1024),strides=(1,1,512),pool_type='AVG'))
dnn.append(ops.BatchNorm(dnn[-1],[0,3]))


dnn.append(layers.Conv2DPool(dnn[-1],[(32,3,3),{'b':None}],
                                [[0,2,3]],[0],[(1,2,4)]))

dnn.append(layers.Conv2DPool(dnn[-1],[(64,3,3),{'b':None}],
                                  [[0,2,3]],[0],[(1,2,4)]))

dnn.append(my_layer(dnn[-1],[1080,{'b':None}],
                                    [[0]],[0]))

dnn.append(ops.Dense(dnn[-1],units=dataset.n_classes))



# Loss and Optimizer
#-------------------

# Compute some quantities that we want to keep track and/or act upon
loss     = crossentropy_logits(p=dataset.labels,q=dnn[-1])
accuracy = accuracy(labels=dataset.labels,predictions=dnn[-1])
auc      = AUC(dataset.labels,tf.nn.softmax(dnn[-1])[:,1])


# we aim at minimizing the loss, so we create the optimizer (Adam in this case)
# with a stepwise learning rate, the minimizer is the operation that applies
# the changes to the model parameters, we also specify that this process
# should also include some possible network dependencies present in UPDATE_OPS

optimizer = sknet.optimize.Adam(loss,0.01,params=dnn.params)
minimizer = tf.group(optimizer.updates+dnn.updates)

# Workers
#---------

work1 = sknet.Worker(op_name='minimizer',context='train_set',op=minimizer, 
        instruction='execute every batch', deterministic=False,
        sampling='random',description='minimize')

work2 = sknet.Worker(op_name='loss',context='train_set',op=loss,
        instruction='save & print every 30 batch', deterministic=False,
        description='saving the loss every 30 batches',sampling='random')

work3 = sknet.Worker(op_name='AUC',context='train_set',op=auc,
        instruction='execute every batch and save & print & average', 
        deterministic=False, description='standard classification accuracy')


queue = sknet.Queue((work1+work2+work3,
                work3.alter(context='valid_set',deterministic=True)))
# Pipeline
#---------

# the pipeline is assembling all the components for executing the program,
# the dataset, the workers and the linkage representing what missing values
# from the network have to be searched for in the dataset 
# (for example, the labels)

workplace = sknet.utils.Workplace(dnn,dataset=dataset)

# will fit the model for 50 epochs and return the gathered op
# outputs given the above definitions

output = workplace.execute_queue(queue,repeat=2000,filename='test.h5',save_period=20)


