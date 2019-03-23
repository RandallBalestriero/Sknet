from ranet import dataset
from ranet.dataset import preprocessing
from ranet.utils import schedules,trainer
from ranet import model,utils

import os
# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gzip
import numpy as np

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

                # Model and trainer
                #------------------
                batch_size = 64
                model      = model_(input_shape=[batch_size]+list(dataset_.spatial_shape),classes=dataset_.classes)
                name       = dataset_.name+preprocessing_.name+lr_schedule.name+model.name
                trainer    = utils.trainer.Trainer(model,lr_schedule, display_name = name)

                # Training
                train_loss,valid_accu,test_accu,lrs = trainer.fit(train_set,valid_set,test_set,n_epochs=120)

                np.savez_compressed(name,
                        train_loss=train_loss,
                        valid_accu=valid_accu,
                        test_accu=test_accu,
                        lrs=lrs)




