from ranet import dataset
from ranet.dataset import preprocessing
from ranet.utils import schedules,trainer
from ranet import models

print(dataset)

# Data Loading
#-------------
train_set,valid_set,test_set = dataset.mnist.load(seed=10)

# Pre-processing
#---------------
preprocessing = preprocessing.zca_whitening(eps=0.0001)

train_set[0] = preprocessing.fit_transform(train_set[0])
valid_set[0] = preprocessing.transform(valid_set[0])
test_set[0]  = preprocessing.transform(test_set[0])

# Learning rate schedule
lr_schedule = schedule.linear(init_lr=0.0001)

# Model and trainer
batch_size = 64
model      = model.smallcnn(input_shape=[batch_size,1,28,28],n_classes=10)
trainer    = utils.trainer.Trainer(model)


# Training
train_loss,valid_accu,test_accu,lrs = trainer.fit(train_set,valid_set,test_set,n_epochs=20)



