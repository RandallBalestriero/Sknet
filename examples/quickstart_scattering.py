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
from sknet import ops,layers
from sknet.utils import flatten
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

dataset = sknet.dataset.load_freefield1010(n_samples=2000,subsample=6)
dataset['signals/train_set']/=dataset['signals/train_set'].max(2,keepdims=True)

if "test_set" not in dataset.sets:
    dataset.split_set("train_set","test_set",0.33)


dataset.create_placeholders(batch_size=5,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                       'valid_set':BatchIterator('continuous'),
                       'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

dnn       = sknet.Network(name='simple_model')

dnn.append(sknet.ops.HermiteSplineConv1D(dataset.signals,
                   J=5, Q=6, K=15, init='random', complex=False,
                   hilbert=False,tied_knots=True,stride=32,
                   tied_weights=True, trainable_filters=True,
                   trainable_knots=False,trainable_scales=False,n_conv=5))
dnn.append(sknet.ops.BatchNorm(dnn[-1],[0,2]))
dnn.append(sknet.ops.Activation(dnn[-1],0.01))
dnn.append(sknet.ops.HermiteSplineConv1D(dnn[-1],
                   J=5, Q=2, K=15, init='random', complex=False,
                   hilbert=False,tied_knots=True,stride=32,
                   tied_weights=True, trainable_filters=True,
                   trainable_knots=False,trainable_scales=False,n_conv=5))
dnn.append(sknet.ops.BatchNorm(dnn[-1],[0,3]))
dnn.append(sknet.ops.Activation(dnn[-1],0.01))
dnn.append(sknet.ops.Pool1D(dnn[-3],1024,pool_type='AVG'))
dnn.append(sknet.ops.Pool1D(dnn[-2],1024,pool_type='AVG'))
dnn.append(sknet.ops.Concat([flatten(dnn[-2]),flatten(dnn[-1])],1))
dnn.append(sknet.ops.Dense(dnn[-1],256))
dnn.append(sknet.ops.BatchNorm(dnn[-1],[0]))
dnn.append(sknet.ops.Activation(dnn[-1],0.01))
dnn.append(sknet.ops.Dense(dnn[-1],2))


prediction = dnn[-1]
GREEDY = 0
if GREEDY:
    reconstruction1 = sknet.losses.MSE(dataset.signals,dnn[0].backward(dnn[2]))
    reconstruction2 = sknet.losses.MSE(dnn[2],dnn[3].backward(dnn[5]))
    reconstruction = reconstruction1+reconstruction2
else:
    reconstruction = sknet.losses.MSE(dataset.signals,dnn[:6].backward(dnn[5]))


loss    = sknet.losses.crossentropy_logits(p=dataset.labels,q=prediction)\
                + reconstruction*0.0001
accu    = sknet.losses.accuracy(dataset.labels,prediction)

B         = dataset.N_BATCH('train_set')
lr        = sknet.schedules.PiecewiseConstant(0.002,
                                    {100*B:0.002,200*B:0.001,250*B:0.0005})
optimizer = sknet.optimizers.Adam(loss,lr,params=dnn.variables(trainable=True))
minimizer = tf.group(optimizer.updates+dnn.updates)

# Workers
#---------

min1  = sknet.Worker(name='minimizer',context='train_set',op=[minimizer,loss],
        deterministic=False, period=[1,1],verbose=[0,3])

accu1 = sknet.Worker(name='accu',context='test_set', op=accu,
        deterministic=True, transform_function=np.mean,verbose=1)

queue = sknet.Queue((min1,accu1))

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_queue(queue,repeat=350)


