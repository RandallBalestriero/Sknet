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



# Data Loading
#-------------

dataset = sknet.dataset.load_cifar10()
#dataset["images"]["train_set"]-=dataset["images"]["train_set"].mean((2,3),keepdims=True)
#dataset["images"]["test_set"]-=dataset["images"]["test_set"].mean((2,3),keepdims=True)
dataset["images"]["train_set"]/=255#dataset["images"]["train_set"].max((2,3),keepdims=True)
dataset["images"]["test_set"]/=255#dataset["images"]["test_set"].max((2,3),keepdims=True)


#dataset.split_set("train_set","valid_set",0.15)
#dataset.preprocess(sknet.dataset.Standardize,data="images",axis=[0])
dataset.create_placeholders(batch_size=64,
        iterators_dict={'train_set':sknet.dataset.BatchIterator("random_all"),
            'test_set':sknet.dataset.BatchIterator('random_all')})

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape
NN_func = tf.identity
#layers = [sknet.layer.Input(batch_size=64,dataset=dataset, input=None)]


layers=[sknet.layer.Conv2D(dataset.images,(64,3,3))]
layers.append(sknet.layer.BatchNorm(layers[-1],[0]))
layers.append(sknet.layer.Activation(layers[-1],0))

layers.append(sknet.layer.Conv2D(layers[-1],(128,3,3)))
layers.append(sknet.layer.BatchNorm(layers[-1],[0]))
layers.append(sknet.layer.Activation(layers[-1],0))
layers.append(sknet.layer.Pool(layers[-1],(1,2,2)))

layers.append(sknet.layer.Conv2D(layers[-1],(192,3,3)))
layers.append(sknet.layer.BatchNorm(layers[-1],[0]))
layers.append(sknet.layer.Activation(layers[-1],0))
layers.append(sknet.layer.Pool(layers[-1],(1,2,2)))

layers.append(sknet.layer.Dense(layers[-1],units=300))
layers.append(sknet.layer.BatchNorm(layers[-1],[0]))
layers.append(sknet.layer.Activation(layers[-1],0))
layers.append(sknet.layer.Dense(layers[-1],units=dataset.n_classes))


network = sknet.network.Network(layers,name='-model(cnn.base)')


# Loss and Optimizer
#-------------------

# Compute some quantities that we want to keep track and/or act upon
loss     = crossentropy_logits(p=dataset.labels,q=network[-1])
accuracy = accuracy(labels=dataset.labels,predictions=network[-1])

# we aim at minimizing the loss, so we create the optimizer (Adam in this case)
# with a stepwise learning rate, the minimizer is the operation that applies
# the changes to the model parameters, we also specify that this process
# should also include some possible network dependencies present in UPDATE_OPS

#learning_rate_schedule = schedule.stepwise({0:0.05})
#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
optimizer = sknet.optimize.Adam(loss,0.01)#schedule = learning_rate_schedule,
minimizer = tf.group(optimizer.updates+tf.get_collection(tf.GraphKeys.UPDATE_OPS))


# Workers
#---------


work1 = sknet.Worker(op_name='minimizer',context='train_set',op=minimizer, 
        instruction='execute every batch', deterministic=False,
        sampling='random',description='minimize')


work2 = sknet.Worker(op_name='loss',context='train_set',op=loss,
        instruction='save & print every 30 batch', deterministic=False,
        description='saving the loss every 30 batches',sampling='random')

work3 = sknet.Worker(op_name='accuracy',context='test_set',op=accuracy,
        instruction='execute every batch and save & print & average', 
        deterministic=True)

#work4 = sknet.Worker(op_name='accuracy',context='valid_set',op=accuracy,
#        instruction='execute every batch and save & print & average', 
#        deterministic=True, description='standard classification accuracy')

work5 = sknet.Worker(op_name='accuracy',context='train_set',op=accuracy,
        instruction='execute every batch and save & print & average', 
        deterministic=False, description='standard classification accuracy')



# Pipeline
#---------

# the pipeline is assembling all the components for executing the program,
# the dataset, the workers and the linkage representing what missing values
# from the network have to be searched for in the dataset 
# (for example, the labels)

workplace = sknet.utils.Workplace(network,dataset=dataset)

# will fit the model for 50 epochs and return the gathered op
# outputs given the above definitions

output = workplace.execute_queue((work1+work2+work5,work3),repeat=2000)
#sknet.to_file(output,'test.h5','w')



