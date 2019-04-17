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

# Data Loading
#-------------
dataset = sknet.dataset.load_cifar10()
dataset.split_set("train_set","test_set",0.25)
dataset.split_set("train_set","valid_set",0.15)

dataset.preprocess(sknet.dataset.Standardize,data="images",axis=[0])

dataset.create_placeholders(batch_size=64,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                        'valid_set':BatchIterator('continuous'),
                        'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape
K = 8
my_layer = layers.custom_layer(ops.Dense,ops.BatchNorm,ops.Activation)

dnn = sknet.network.Network(name='simple_model')

dnn.append(ops.RandomCrop(dataset.images,(28,28)))

dnn.append(layers.Conv2D(dnn[-1],[(64,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                            [0.01]))
dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

dnn.append(layers.Conv2DPool(dnn[-2],[(192,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                                [0.01],[(1,2,2)]))
dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

dnn.append(layers.Conv2D(dnn[-2],[(256,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                                [0.01]))
dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

dnn.append(layers.Conv2DPool(dnn[-2],[(512,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                                [0.01],[(1,2,2)]))
dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

dnn.append(layers.Conv2D(dnn[-2],[(512,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                                [0.01]))
dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

dnn.append(layers.Conv2DPool(dnn[-2],[(712,3,3),{'b':None}],[[0,2,3]],
                                                [0.01],[(1,2,2)]))
dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))






predictions = [dnn[i*2] for i in range(1,7)]

# Loss and Optimizer
#-------------------
# Compute some quantities that we want to keep track and/or act upon
losses = [crossentropy_logits(p=dataset.labels,q=pred)
                        for pred in predictions]

accus = [accuracy(dataset.labels,pred)
                                for pred in predictions]


optimizers    = list()
B             = dataset.N('train_set')//64
learning_rate = sknet.optimize.PiecewiseConstant(0.01,
                                    {80*B:0.005,120*B:0.001,160*B:0.0001})

for i,l in enumerate(losses):
    optimizers.append(Adam(l,learning_rate,params=dnn[i*2+1:(i+1)*2+1].params))

minimizer = tf.group(sum([opt.updates for opt in optimizers],[])+dnn.updates)

# Workers
#---------

min1 = sknet.Worker(op_name='minimizer',context='train_set',op=minimizer, 
        instruction='execute every batch', deterministic=False)

loss1 = sknet.Worker(op_name='loss',context='train_set',op=tf.stack(losses,-1),
        instruction='save & print every 100 batch', deterministic=False)

op_accu = tf.reshape(tf.stack(accus),(1,6))

accus1 = sknet.Worker(op_name='prediction',context='test_set',
        op=op_accu,
        instruction='execute every batch and save & print & average', 
        deterministic=True, description='standard classification accuracy')

queue1 = sknet.Queue((min1+loss1,accus1.alter(context='valid_set'),accus1))

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)

workplace.execute_queue(queue1,repeat=200)

predictions = pred1.data[-1]


#for i in range(5):
#    fig = plt.figure(figsize=(12,4))
#    ax  = fig.gca(projection='3d')
#    ax.plot_trisurf(predictions[::16,0], predictions[::16,1], 
#                predictions[::16,2+i],antialiased=True)

#    ax.set_title(str(i))

plt.show()


#sknet.to_file(output,'test.h5','w')



