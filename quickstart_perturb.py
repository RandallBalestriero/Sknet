import sknet
from sknet import dataset
from sknet.utils import plotting
import pylab as pl
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_and_plot(input_shape, option,dataset):

    # create our custom deep net just for data augmentation
    # purposes, thus omiting all the following layers
    layers      = [sknet.layer.Input(input_shape,data_format=dataset["data_format"])]
    if option=="crop":
        layers.append(sknet.layer.RandomCrop(layers[-1], crop_shape=(26,26)))
        name = "Crop2626"
    elif option=='rot':
        layers.append(sknet.layer.RandomRot90(layers[-1]))
        name = "Rot90"
    elif option=='reverse':
        layers.append(sknet.layer.RandomAxisReverse(layers[-1],axis=3))
        name = "AxisReverse"
    elif option=='gaussian':
        layers.append(sknet.layer.Gaussian(layers[-1]))
        name = "Gaussian"
    elif option=='uniform':
        layers.append(sknet.layer.Uniform(layers[-1]))
        name = "Uniform"
    elif option=='uniformmul':
        layers.append(sknet.layer.Uniform(layers[-1],noise_type='multiplicative'))
        name = "UniformMul"
    elif option=='dropout':
        layers.append(sknet.layer.Dropout(layers[-1]))
        name = "Dropout"



    # wrap the layers as a deep net model
    model = sknet.network.Network(layers,name=name)

    # set the trainer with this model to create the tensorflow
    # working environment and gather all the pieces together
    # we use the DummyTrainer to get the minimal function as 
    # we won't train etc
    trainer    = sknet.utils.trainer.DummyTrainer(model)

    # get a batch of images, we use the same image for the whole
    # batch to show the effect of the data augmentation on the same image
    batch_images = dataset["train_set"][0][[0]].repeat(input_shape[0],0)
    batch = trainer.session.run(trainer.output, feed_dict={
                    trainer.network.deterministic:False, 
                    trainer.input:batch_images})

    pl.figure(figsize=(20,2))
    pl.subplot(1,10,1)
    plotting.imshow(dataset["train_set"][0][0])
    pl.title('original')
    for cpt in range(9):
        pl.subplot(1,10,2+cpt)
        plotting.imshow(batch[cpt])

    # Reduce side margins
    pl.tight_layout()
    pl.suptitle(name)
    pl.savefig('./sknet/docs/source/_static/augmentation_'+name+'.png')
    pl.close()


def main():
    # Load CIFAR10 dataset for demonstration
    # by default data_format='NCHW'
    cifar10 = dataset.cifar10()

    # Import the data
    cifar10.load()
    # we standardize the data to not worry
    # about the std and sclaes of the noise layers
    # compared to the scale of the data
    cifar10.preprocess(sknet.dataset.Standardize)

    # Set input shape
    batch_size  = 64
    input_shape = [batch_size]+list(cifar10["datum_shape"])
    
    # Loop over all the perturb layer options
    options = ['rot', 'crop', 'reverse', 'gaussian', 'uniform', 
                        'uniformmul', 'dropout']
    for option in options:
        build_and_plot(input_shape, option, cifar10)


if __name__=='__main__':
    main()

