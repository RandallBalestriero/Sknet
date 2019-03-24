from sknet import dataset
from sknet.dataset import preprocessing
from sknet.utils import schedules,trainer
from sknet import model,utils
import matplotlib
matplotlib.use('Agg')
import os
# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gzip
import numpy as np
import pylab as pl
import time


datasets       = [dataset.cifar10,dataset.cifar100]
preprocessings = [preprocessing.identity(),
                    preprocessing.zca_whitening(eps=0.0001),
                    preprocessing.zca_whitening(eps=0.0001)]

models         = [model.cnn.base]

lr_schedules   = [schedules.exponential(init_lr=0.001,step=0.95,adaptive=True),
                    schedules.constant(init_lr=0.001),
                    schedules.exponential(init_lr=0.001,step=0.95),
                    schedules.stepwise({0:0.001,50:0.0001,100:0.00005})] 



for dataset_ in datasets:

    # Data Loading
    #-------------
    train_set,valid_set,test_set = dataset_.load(seed=10)

    for preprocessing_ in preprocessings:

        # Pre-processing
        #---------------
        train_set[0] = preprocessing_.fit_transform(train_set[0])
        valid_set[0] = preprocessing_.transform(valid_set[0])
        test_set[0]  = preprocessing_.transform(test_set[0])

        for lr_schedule in lr_schedules:

            # Learning rate schedule
            #-----------------------
            schedule = lr_schedule.init()

            for model_ in models:
                
                t_ = time.time()

                # Model and trainer
                #------------------
                batch_size = 64
                model      = model_(input_shape=[batch_size]+list(dataset_.image_shape),classes=dataset_.classes,data_format=dataset_.data_format)
                name       = dataset_.name+preprocessing_.name+lr_schedule.name+model.name
                trainer    = utils.trainer.Trainer(model,lr_schedule, display_name = name)

                # Training
                train_loss,valid_accu,test_accu,lrs = trainer.fit(train_set,valid_set,test_set,n_epochs=120)
                
                best_accu = test_accu[np.argmax(valid_accu)]

                pl.figure(figsize=(20,4))
                pl.subplot(131)
                pl.plot(np.concatenate(train_loss,0))
                pl.title('Train loss')
                pl.subplot(132)
                pl.plot(valid_accu,color='r')
                pl.plot(test_accu,color='k')
                pl.title('Valid (red) and Test (black) accuracy')
                pl.subplot(133)
                pl.plot(lrs)
                pl.title('Learning Rates')
                pl.suptitle(name+' {:1f}, in {}min.'.format(best_accu*100,np.int32((time.time()-t_)/60)))
                pl.savefig(name+'.png')
#np.savez_compressed(name,
#                        train_loss=train_loss,
#                        valid_accu=valid_accu,
#                        test_accu=test_accu,
#                        lrs=lrs)




