import sknet
from sknet.optimize import Adam
from sknet.optimize.loss import *
from sknet.optimize import schedule
import matplotlib
matplotlib.use('Agg')
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import numpy as np
import pylab as pl
import time
import tensorflow as tf



# Data Loading
#-------------

dataset = sknet.dataset.load_svhn()
dataset.split_set("train_set","valid_set",0.15)
dataset.preprocess(sknet.dataset.Standardize,data="images")


# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape
input_shape = [64]+list(dataset.datum_shape)


layers = [sknet.layer.Input(batch_size=64,dataset=dataset, input=None)]
layers.append(sknet.layer.Conv2D(layers[-1],filters=(32,3,3)))
layers.append(sknet.layer.BatchNormalization(layers[-1],[0,2,3],filters=(32,3,3)))

layers.append(sknet.layer.Pool(layers[-1],(2,2)))
layers.append(sknet.layer.Activation(layers[-1],0))
layers.append(sknet.layer.Conv2D(layers[-1],filters=(64,3,3)))
layers.append(sknet.layer.BatchNormalization(layers[-1],[0,2,3],filters=(32,3,3)))

layers.append(sknet.layer.Pool(layers[-1],(2,2)))
layers.append(sknet.layer.Activation(layers[-1],0))
layers.append(sknet.layer.Conv2D(layers[-1],filters=(128,3,3)))
layers.append(sknet.layer.BatchNormalization(layers[-1],[0,2,3],filters=(32,3,3)))

layers.append(sknet.layer.Activation(layers[-1],0))
layers.append(sknet.layer.Conv2D(layers[-1],filters=(256,1,1)))
layers.append(sknet.layer.BatchNormalization(layers[-1],[0,2,3],filters=(32,3,3)))

layers.append(sknet.layer.Pool(layers[-1],(8,1,1)))
layers.append(sknet.layer.Dense(layers[-1],units=dataset.n_classes))

network = sknet.network.Network(layers,name='-model(cnn.base)')

# Loss and Optimizer
#-------------------

# Compute some quantities that we want to keep track and/or act upon
loss     = crossentropy_logits(p=None,q=network[-1])
accuracy = accuracy(labels=None,predictions=network[-1])

# we aim at minimizing the loss, so we create the optimizer (Adam in this case)
# with a stepwise learning rate, the minimizer is the operation that applies
# the changes to the model parameters, we also specify that this process
# should also include some possible network dependencies present in UPDATE_OPS
learning_rate_schedule = schedule.stepwise({0:0.001,5000:0.0005,100000:0.0001})
optimizer = Adam(schedule = learning_rate_schedule,
                    dependencies=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
minimizer = optimizer.minimize(loss)



# Workers
#---------


work1 = sknet.Worker(op_name='minimizer',context='train_set',op=minimizer, 
        instruction='execute every batch', deterministic=False, 
        description=optimizer.description)

work2 = sknet.Worker(op_name='loss',context='train_set',op=loss,
        instruction='save & print every 30 batch', deterministic=False,
        description='saving the loss every 30 batches')

work3 = sknet.Worker(op_name='accuracy',context='test_set',op=accuracy,
        instruction='execute every batch and save & print & average', 
        deterministic=True)

work4 = sknet.Worker(op_name='accuracy',context='valid_set',op=accuracy,
        instruction='execute every batch and save & print & average', 
        deterministic=True, description='standard classification accuracy')

work5 = sknet.Worker(op_name='accuracy',context='train_set',op=accuracy,
        instruction='execute every batch and save & print & average', 
        deterministic=True, description='standard classification accuracy')



# Pipeline
#---------

# the pipeline is assembling all the components for executing the program,
# the dataset, the workers and the linkage representing what missing values
# from the network have to be searched for in the dataset 
# (for example, the labels)

workplace = sknet.utils.Workplace(network,dataset=dataset,
                        linkage={network[0].name:"images",
                        loss.p.name:"labels",
                        accuracy.labels.name:"labels"})


# will fit the model for 50 epochs and return the gathered op
# outputs given the above definitions

output = workplace.execute_queue((work1+work2+work5,work3,work4),repeat=10)
#sknet.to_file(output,'test.h5','w')



