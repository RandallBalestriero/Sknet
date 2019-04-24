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
noise_l   = np.float32(sys.argv[-2])
gamma     = np.float32(sys.argv[-1])

dnn.append(ops.RandomAxisReverse(dataset.images,axis=[-1]))
dnn.append(ops.RandomCrop(dnn[-1],(28,28),seed=10))
dnn.append(ops.GaussianNoise(dnn[-1],noise_type='additive',
                        sigma=noise_l))

dnn.append(ops.Concat([dnn[-1],dnn[-2]],axis=0))

dnn.append(layers.Conv2D(dnn[-1],[(64,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                            [0.01]))
dnn.append(layers.Conv2D(dnn[-1],[(128,3,3),{'b':None,'pad':'same'}],
                                    [[0,2,3]], [0.01]))
dnn.append(layers.Conv2DPool(dnn[-1],[(128,3,3),{'b':None}],
                                    [[0,2,3]],[0.01],[(1,2,2)]))

dnn.append(layers.Conv2D(dnn[-1],[(192,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                                [0.01]))
dnn.append(layers.Conv2D(dnn[-1],[(192,3,3),{'b':None}],[[0,2,3]],
                                                [0.01]))
dnn.append(layers.Conv2DPool(dnn[-1],[(192,3,3),{'b':None,'pad':'same'}],
                                    [[0,2,3]],[0.01],[(1,2,2)]))

dnn.append(layers.Conv2D(dnn[-1],[(192,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                                [0.01]))
dnn.append(layers.Conv2D(dnn[-1],[(192,3,3),{'b':None,'pad':'same'}],[[0,2,3]],
                                                [0.01]))
dnn.append(layers.Conv2D(dnn[-1],[(192,1,1),{'b':None}],[[0,2,3]],
                                                [0.01]))
dnn.append(ops.Pool(dnn[-1],(1,-1,-1),pool_type='AVG'))
dnn.append(ops.Dense(dnn[-1],10))

# Quantities
#-----------

A = tf.gradients([dnn[-1]]*10,[dnn[3]]*10,
                [tf.ones((64,1))*tf.one_hot(i,10) for i in range(10)])

prediction = dnn[-1]

# Loss and Optimizer
#-------------------

loss    = crossentropy_logits(p=dataset.labels,q=prediction[:32])
hessian = tf.add_n([sknet.optimize.loss.SSE(a[:32],a[32:])
                        for a in A])#/np.float32(len(A))
accu    = accuracy(dataset.labels,prediction[:32])

B         = dataset.N('train_set')//32
lr        = sknet.optimize.PiecewiseConstant(0.005,
                                    {100*B:0.003,200*B:0.001,250*B:0.0005})
optimizer = Adam(loss+hessian*gamma,lr,params=dnn.params)
minimizer = tf.group(optimizer.updates+dnn.updates)


# Workers
#---------

min1 = sknet.Worker(op_name='minimizer',context='train_set',op=minimizer,
        instruction='execute every batch', deterministic=False)

loss = tf.expand_dims(tf.stack([loss,hessian]),0)
loss_worker = sknet.Worker(op_name='loss',context='train_set',op= loss,
        instruction='save & print every 100 batch', deterministic=False)

accu = tf.expand_dims(tf.stack([accu,hessian]),0)
accu_worker = sknet.Worker(op_name='accu',context='test_set', op=accu,
        instruction='execute every batch and save & print & average',
        deterministic=True, description='standard classification accuracy')

queue = sknet.Queue((min1+loss_worker,accu_worker.alter(context='valid_set'),
                            accu_worker))

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_queue(queue,repeat=350,
            filename='cifar10_'+str(noise_l)+'_'+str(gamma)+'.h5',
            save_period=20)



#sknet.to_file(output,'test.h5','w')



