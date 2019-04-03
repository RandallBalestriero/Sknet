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
model       = sknet.network.cnn.base(input_shape=input_shape,
                    n_classes=dataset.n_classes,
                    data_format=dataset.data_format)

# Loss and Optimizer
#-------------------

lr_schedule = sknet.schedule.stepwise({0:0.001,50:0.0001,100:0.00005})
loss        = sknet.loss.classification(model[-1])
test_loss   = sknet.loss.accuracy(model[-1])
optimizer   = sknet.optimizer.Adam()

# Pipeline
#---------

pipeline    = sknet.utils.Pipeline(model,dataset,lr_schedule=lr_schedule,
                            loss=loss,optimizer=optimizer,test_loss=test_loss)

pipeline.link([[pipeline.network[0].observation,"images"],
                [pipeline.network[-1].observation,"labels"]])

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



