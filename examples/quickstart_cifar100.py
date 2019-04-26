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

dnn       = sknet.network.Network(name='simple_model')

dnn.append(ops.RandomAxisReverse(dataset.images,axis=[-1]))
dnn.append(ops.RandomCrop(dnn[-1],(28,28),seed=10))
dnn.append(ops.GaussianNoise(dnn[-1],noise_type='additive',sigma=0.05))

dnn = sknet.network.Resnet(dnn,dataset.n_classes,D=4,W=2)
print(dataset.n_classes)
prediction = dnn[-1]

print(prediction.get_shape().as_list())
loss    = crossentropy_logits(p=dataset.labels,q=prediction)
accu    = accuracy(dataset.labels,prediction)

B         = dataset.N('train_set')//64
lr        = sknet.optimize.PiecewiseConstant(0.005,
                                    {100*B:0.003,200*B:0.001,250*B:0.0005})
optimizer = Adam(loss,lr,params=dnn.params)
minimizer = tf.group(optimizer.updates+dnn.updates)

# Workers
#---------

min1 = sknet.Worker(op_name='minimizer',context='train_set',op=minimizer,
        instruction='execute every batch', deterministic=False)

loss_worker = sknet.Worker(op_name='loss',context='train_set',op= loss,
        instruction='save & print every 100 batch', deterministic=False)

accu_worker = sknet.Worker(op_name='accu',context='test_set', op=accu,
        instruction='execute every batch and save & print & average',
        deterministic=True, description='standard classification accuracy')

queue = sknet.Queue((min1+loss_worker,accu_worker.alter(context='valid_set'),
                            accu_worker))

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_queue(queue,repeat=350,
            filename='cifar100_classification.h5',
            save_period=50)



#sknet.to_file(output,'test.h5','w')



