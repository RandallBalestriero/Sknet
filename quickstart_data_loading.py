from sknet import dataset
from sknet.utils import plotting
import pylab as pl

# Put the dataset functions into a list s.t. dataset_list[0].load() 
# loads the dataset 0
dataset_list = [dataset.mnist(),
                dataset.fashionmnist(),
                dataset.svhn(),
                dataset.cifar10(),
                dataset.cifar100()]

# Save number of dataset
n_dataset    = len(dataset_list)

# Initialize the counter for subplot
cpt          = 1

# Loop over the dataset_list to load them (download them if necessary)
# and display the first 10 images
pl.figure(figsize=(20,n_dataset*2))
for dataset in dataset_list:
    train_set,valid_set,test_set = dataset.load(seed=10)
    for im,label in zip(train_set[0][:10],train_set[1][:10]):
        pl.subplot(n_dataset,10,cpt)
        pl.title(str(label))
        plotting.imshow(im)
        cpt+=1

# Reduce side margins
pl.tight_layout()

pl.savefig('./sknet/docs/source/user/test_loading.png')

