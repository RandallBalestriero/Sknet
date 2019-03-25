from sknet.dataset import mnist,svhn,cifar10,fashionmnist,cifar100
from sknet.dataset import preprocess
from sknet.utils import plotting

import pylab as pl

# Put the preprocessing function into a list
preprocessing_list = [preprocessing.standardize(eps=0.0001),
                        preprocessing.zca_whitening(eps=0.0001)]

# Save number of preprocessing
n_preprocessing = len(preprocessing_list)

# Put the dataset functions into a list s.t. dataset_list[0].load() 
# loads the dataset 0
dataset_list = [mnist,fashionmnist,svhn,cifar10,cifar100]
dataset_name = ['mnist','fashionmnist','svhn','cifar10','cifar100']

# Save number of dataset
n_dataset    = len(dataset_list)

for dataset,dataset_n in zip(dataset_list,dataset_name):
    pl.figure(figsize=(20,n_preprocessing*2))
    train_set,valid_set,test_set = dataset.load(seed=10)
    # Initialize the counter for subplot
    cpt          = 1
    for i,im in enumerate(train_set[0][:10]):
        pl.subplot(n_preprocessing+1,10,cpt)
        plotting.imshow(im)
        cpt+=1
        if(i==4):
            pl.title('Original Data')
    for preprocessing in preprocessing_list:
        preprocessing.fit(train_set[0])
        images = preprocessing.transform(train_set[0][:10])
        for i,im in enumerate(images):
            pl.subplot(n_preprocessing+1,10,cpt)
            plotting.imshow(im)
            cpt+=1
            if(i==4):
                pl.title(preprocessing.name)
    pl.tight_layout()
    pl.savefig('test_preprocessing_'+dataset_n+'.png')

