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
N = 400
dataset = sknet.dataset.load_chirp2D(N)
#dataset.split_set("train_set","test_set",0.25)
#dataset.split_set("train_set","valid_set",0.15)

dataset.preprocess(sknet.dataset.Standardize,data="input",axis=[0])

dataset.create_placeholders(batch_size=100,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                        'valid_set':BatchIterator('continuous'),
                        'test_set':BatchIterator('continuous')},device="/GPU:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape
K = 8
my_layer = layers.custom_layer(ops.Dense,ops.BatchNorm,ops.Activation)

dnn = sknet.network.Network(name='simple_model')

dnn.append(my_layer(dataset.input,[K,{'b':None}],[0],[0.01]))
dnn.append(my_layer(dnn[-1],[1,{'b':None}],[0],[tf.nn.tanh]))
for i in range(4):
    dnn.append(my_layer(dnn[-2],[128,{'b':None}],[0],[0.01]))
    dnn.append(my_layer(dnn[-1],[1,{'b':None}],[0],[tf.nn.tanh]))

predictions = [dnn[i*2+1] for i in range(5)]

# Loss and Optimizer
#-------------------
# Compute some quantities that we want to keep track and/or act upon
losses = [MSE(target=dataset.output,prediction=tf.squeeze(pred)) 
                        for pred in predictions]

optimizers = [sknet.optimize.Adam(l,0.001,params=dnn[i*2:(i+1)*2].params) 
                            for i,l in enumerate(losses)]

minimizer = tf.group(sum([opt.updates for opt in optimizers],[])+dnn.updates)

# Workers
#---------

min1 = sknet.Worker(op_name='minimizer',context='train_set',op=minimizer, 
        instruction='execute every batch', deterministic=False)

loss1 = sknet.Worker(op_name='loss',context='train_set',op=tf.stack(losses,-1),
        instruction='save & print every 1000 batch', deterministic=False)

pred1 = sknet.Worker(op_name='prediction',context='train_set',
        op=tf.concat([dataset.input]+predictions,1),
        instruction='save every batch', 
        deterministic=True, description='standard classification accuracy')

queue1 = sknet.Queue((min1+loss1,pred1))

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)

workplace.execute_queue(queue1,repeat=20)

predictions = pred1.data[-1]


for i in range(5):
    fig = plt.figure(figsize=(12,4))
    ax  = fig.gca(projection='3d')
    ax.plot_trisurf(predictions[::16,0], predictions[::16,1], 
                predictions[::16,2+i],antialiased=True)

    ax.set_title(str(i))

plt.show()


#sknet.to_file(output,'test.h5','w')



