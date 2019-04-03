from sknet import dataset
from sknet.utils import plotting
import pylab as pl

# Put the dataset functions into a list to loop over
dataset_list = [dataset.load_mnist,
                dataset.load_fashionmnist,
                dataset.load_svhn,
                dataset.load_cifar10,
                dataset.load_cifar100,
                dataset.load_stl10,
                dataset.load_warblr,
                dataset.load_freefield1010]

# Loop over the dataset_list to load them (download them if necessary)
# and display the first 20 images
for dataset_func in dataset_list:

    # Load data
    dataset=dataset_func()

    # Create the figure
    pl.figure(figsize=(20,4))

    # distinguish the cifrar100 case as it has coarse and fine labels
    if dataset.name=='cifar100':
        images        = dataset['images']['train_set'][:20]
        fine_labels   = dataset['labels']['train_set'][:20]
        coarse_labels = dataset['coarse_labels']['train_set'][:20]
        for im,coarse,fine,cpt in zip(images,coarse_labels,fine_labels,range(20)):
            pl.subplot(2,10,cpt+1)
            pl.title(str(coarse)+', '+str(fine)+': '\
                    +dataset.classes[fine])
            plotting.imshow(im)
    else:
        labels = dataset['labels']['train_set'][:20]
        if 'images' in dataset.variables:
            images = dataset['images']['train_set'][:20]
        else:
            images = dataset['signals']['train_set'][:20]
        for im,label,cpt in zip(images,labels,range(20)):
            pl.subplot(2,10,cpt+1)
            if len(im)>2:
                plotting.imshow(im)
            else:
                signal = pl.squeeze(im)
                pl.plot(signal)
                pl.xlim([0,len(signal)])
                pl.xticks([])
                pl.yticks([])
            pl.title(str(label)+": "+dataset.classes[label])

    # Reduce side margins and save fig
    pl.tight_layout()
    pl.savefig('./sknet/docs/source/_static/'+dataset.name+'_samples.png')
    pl.close()
