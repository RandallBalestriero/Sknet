from sknet import dataset
from sknet.utils import plotting
import pylab as pl

# Put the dataset functions into a list to loop over
dataset_list = [dataset.mnist(),
#                dataset.fashionmnist(),
#                dataset.svhn(),
#                dataset.cifar10(),
                dataset.cifar100(),
                dataset.stl10()]

# Loop over the dataset_list to load them (download them if necessary)
# and display the first 20 images
for dataset in dataset_list:

    # Load data
    dataset.load()

    # Create the figure
    pl.figure(figsize=(20,4))

    images = dataset['train_set'][0][:20]
    # distinguish the cifrar100 case as it has coarse and fine labels
    if dataset["name"]=='cifar100':
        fine_labels   = dataset['train_set'][2][:20]
        coarse_labels = dataset['train_set'][1][:20]
        for im,coarse,fine,cpt in zip(images,coarse_labels,fine_labels,range(20)):
            pl.subplot(2,10,cpt+1)
            pl.title(str(coarse)+', '+str(fine)+': '\
                    +dataset['classes'][fine])
            plotting.imshow(im)
    else:
        labels = dataset['train_set'][1][:20]
        for im,label,cpt in zip(images,labels,range(20)):
            pl.subplot(2,10,cpt+1)
            pl.title(str(label)+": "+dataset['classes'][label])
            plotting.imshow(im)

    # Reduce side margins and save fig
    pl.tight_layout()
    pl.savefig('./sknet/docs/source/_static/'+dataset['name']+'_samples.png')
    pl.close()
