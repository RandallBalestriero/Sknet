import sknet
from sknet.dataset import mnist,svhn,cifar10,fashionmnist,cifar100
from sknet.utils import plotting
import pylab as pl

import os
# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


train_set,valid_set,test_set = cifar10.load(seed=10)


#### Rotation Case
batch_size  = 64
input_shape = [batch_size]+list(cifar10.image_shape)
models      = [
        sknet.model.minimal.augmentation(input_shape=input_shape,
        crop=True,crop_size=(26,26),left_right=False,up_down=False,rot90=False,data_format=cifar10.data_format),
        sknet.model.minimal.augmentation(input_shape=input_shape,
        crop=False,crop_size=(26,26),left_right=True,up_down=False,rot90=False,data_format=cifar10.data_format),
        sknet.model.minimal.augmentation(input_shape=input_shape,
        crop=False,crop_size=(26,26),left_right=False,up_down=True,rot90=False,data_format=cifar10.data_format),
        sknet.model.minimal.augmentation(input_shape=input_shape,
        crop=False,crop_size=(26,26),left_right=False,up_down=False,rot90=True,data_format=cifar10.data_format)
        ]
models_names = ['Random Crop (26,26)','Random Left-Right Flip','Random Up-Down Flip','Random Rot90']


for model,model_name in zip(models,models_names):

    trainer    = sknet.utils.trainer.DummyTrainer(model)
    batch      = trainer.session.run(trainer.layers[-1].output,
        feed_dict={trainer.training:True,trainer.x:train_set[0][[0]].repeat(batch_size,0)})

    pl.figure(figsize=(20,2))
    pl.subplot(1,10,1)
    plotting.imshow(train_set[0][0])
    for i in range(9):
        pl.subplot(1,10,2+i)
        plotting.imshow(batch[i])

    # Reduce side margins
    pl.tight_layout()
    pl.suptitle(model_name)
    pl.savefig('augmentation_'+model_name+'.png')
    pl.close()

