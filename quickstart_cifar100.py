from ranet import dataset
from ranet.dataset import preprocessing
from ranet.utils import schedules,trainer
from ranet import model,utils
import matplotlib
matplotlib.use('Agg')
import os
# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gzip
import numpy as np
import pylab as pl
import time



# Data Loading
#-------------
train_set,valid_set,test_set = dataset.cifar100.load(seed=10)

# Pre-processing
#---------------
zca = preprocessing.zca_whitening()
train_set[0] = zca.fit_transform(train_set[0])
valid_set[0] = zca.transform(valid_set[0])
test_set[0]  = zca.transform(test_set[0])





# Learning rate schedule
#-----------------------
lr_schedule = schedules.stepwise({0:0.001,50:0.0001,100:0.00005})
schedule    = lr_schedule.init()



# Model and trainer
#------------------
batch_size = 64
model      = model.cnn.base(input_shape=[batch_size]+list(dataset.cifar100.image_shape),
                classes=dataset.cifar100.classes,
                data_format=dataset.cifar100.data_format)
name       = dataset.cifar100.name+zca.name\
                    +lr_schedule.name+model.name
trainer    = utils.trainer.Trainer(model,lr_schedule, display_name = name)

# Training
#---------
t_ = time.time()
train_loss,valid_accu,test_accu,lrs = trainer.fit(train_set,valid_set,test_set,n_epochs=120)
                
best_accu = test_accu[np.argmax(valid_accu)]

print('Finished Training '+name+'\n in {:1f}s. with test accuracy {}'.format(time.time()-t_,best_accu))



