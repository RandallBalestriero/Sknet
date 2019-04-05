import sknet
import matplotlib
matplotlib.use('Agg')
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pylab as pl
import time
import tensorflow as tf


# Data Loading
#-------------

dataset = sknet.dataset.load_mnist()

# Pre-processing
#---------------

dataset.preprocess(sknet.dataset.Identity,data="images")

# Model
#------

batch_size  = 64
input_shape = [batch_size]+list(dataset.datum_shape)



with tf.device("/device:GPU:0"):
    layers = [sknet.layer.Input(input_shape=input_shape,data_format=dataset.data_format)]
    layers.append(sknet.layer.Dense(layers[-1],units=512))
#    layers.append(tf.layers.dense(layers[-1],1225))
#    layers.append(tf.layers.dense(layers[-1],1225))
#    layers.append(tf.layers.dense(layers[-1],225))
    layers.append(sknet.layer.Dense(layers[-1],units=512))
    layers.append(sknet.layer.Activation(layers[-1],tf.nn.relu))
    layers.append(sknet.layer.Dense(layers[-1],units=128))
    layers.append(sknet.layer.Activation(layers[-1],tf.nn.relu))
    layers.append(sknet.layer.Dense(layers[-1],units=10,data_format='NCHW'))

    layers[-1].add_loss(sknet.optimize.loss.sparse_cross_entropy_logits(p=None,q=layers[-1]))

    network = sknet.network.Network(layers,name='-model(cnn.base)')

    # Loss and Optimizer
    #-------------------

    lr_schedule = sknet.optimize.schedule.stepwise({0:0.001,50:0.0001,100:0.00005})
    test_loss   = network[-1].sparse_cross_entropy_logits0.accuracy
    optimizer   = sknet.optimize.optimizer.Adam()

    # Pipeline
    #---------

    pipeline    = sknet.utils.Pipeline(network,dataset,lr_schedule=lr_schedule,
                    optimizer=optimizer,test_loss=test_loss)


pipeline.link([[pipeline.network[0],"images"],
                [pipeline.network[-1].sparse_cross_entropy_logits0.p,"labels"]])

# Training
#---------

pipeline.fit(n_epochs=20)

exit()
# MANUAL

pipeline._epoch_passive([
            [pipeline.network[0],pipeline.dataset["train_set"][0]]])

for _ in range(20):
    pipeline._epoch_train()
    print(pipeline._test())


#name       = dataset.cifar100.name+zca.name\
#                    +lr_schedule.name+model.name


# Training
#---------
t_ = time.time()
train_loss,valid_accu,test_accu,lrs = trainer.fit(train_set,valid_set,test_set,n_epochs=120)
 
best_accu = test_accu[np.argmax(valid_accu)]

print('Finished Training '+name+'\n in {:1f}s. with test accuracy {}'.format(time.time()-t_,best_accu))



