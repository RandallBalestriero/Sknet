from sknet import dataset
from sknet.utils import plotting
import pylab as pl

# Put the dataset functions into a list s.t. dataset_list[0].load() 
# loads the dataset 0
dataset_list = [dataset.mnist(),
                dataset.fashionmnist(),
                dataset.svhn(),
                dataset.cifar10(),
                dataset.cifar100(),
                dataset.stl10()]

# Loop over the dataset_list to load them (download them if necessary)
# and display the first 10 images

for dataset in dataset_list:
    dataset.load()

    # Create the figure
    pl.figure(figsize=(20,4))

    # Initialize the counter for subplot
    cpt = 1

    # distinguish the cifrar100 case as it has coarse and
    # fine labels
    if dataset["name"]=='cifar100':
        for im,coarse_label,fine_label in zip(dataset['train_set'][0][:20],dataset['train_set'][1][:20],dataset['train_set'][2][:20]):
            pl.subplot(2,10,cpt)
            pl.title(dataset['superclasses'][label]+', '+dataset['classes'][label])
            plotting.imshow(im)
            cpt+=1
    else:
        for im,label in zip(dataset['train_set'][0][:20],dataset['train_set'][1][:20]):
            pl.subplot(2,10,cpt)
            pl.title(dataset['classes'][label])
            plotting.imshow(im)
            cpt+=1

    # Reduce side margins and save
    pl.tight_layout()
    pl.savefig('./sknet/docs/source/_static/'+dataset['name']+'_samples.png')
    pl.close()
