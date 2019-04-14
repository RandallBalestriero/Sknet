import sknet as sk
import numpy as np

np.random.seed(1)

#-------------------------
# DataArray
#-------------------------


# Smart (intelligible) first axis indexing
train_indices = np.arange(1,4)
data = np.random.randn(4,5)

print(data)
#[[ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763]
# [-2.3015387   1.74481176 -0.7612069   0.3190391  -0.24937038]
# [ 1.46210794 -2.06014071 -0.3224172  -0.38405435  1.13376944]
# [-1.09989127 -0.17242821 -0.87785842  0.04221375  0.58281521]]

images = sk.DataArray(data,{'train_indices':train_indices}, 
                name='images', info='RGB digits')
print(images.name)
#'images'


# usual data access
print(images[0])
# [ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763]

# intelligible way, same as images[range(1,3)]
print(images["train_indices"])
#[[-2.3015387   1.74481176 -0.7612069   0.3190391  -0.24937038]
# [ 1.46210794 -2.06014071 -0.3224172  -0.38405435  1.13376944]
# [-1.09989127 -0.17242821 -0.87785842  0.04221375  0.58281521]]

# overloaded operators
print((images+1).partition)
#{'train_indices': array([1, 2, 3])}

print((1+images)["train_indices"])
#[[-1.3015387   2.74481176  0.2387931   1.3190391   0.75062962]
# [ 2.46210794 -1.06014071  0.6775828   0.61594565  2.13376944]
# [-0.09989127  0.82757179  0.12214158  1.04221375  1.58281521]]

# additional feature: same as images[range(1,3)[0]]
print(images["train_indices",0])
#[-2.3015387   1.74481176 -0.7612069   0.3190391  -0.24937038]

# additional feature: same as images[range(1,3)[[0]]]
print(images["train_indices",[0]])
#[[-2.3015387   1.74481176 -0.7612069   0.3190391  -0.24937038]]

# additional feature: same as images[range(1,3)[[0,1]]]
print(images["train_indices",range(2)])
#[[-2.3015387   1.74481176 -0.7612069   0.3190391  -0.24937038]
# [ 1.46210794 -2.06014071 -0.3224172  -0.38405435  1.13376944]]

# 60-94
###############
# sknet.dataset
###############


mnist = sk.dataset.load_mnist()
# will print:
# Loading mnist
# Dataset mnist loaded in 0.73 s

## bunch of attributes:

mnist.data_format
# 'NCHW' (as left by default in load_mnist
mnist.datum_shape
# (1,28,28)
mnist.n_channels
# 1

## bunch of variables with names 'images' and 'labels as given by
## the attribute

mnist.variables
# ['images','labels']

# each variable is further split into at least 
# a train_set, and possibly more s.a.
# train_set, valid_set, test_set,sunlabeled_set, ...
# but for each variables the sets are the same

#similar to mnist[var].sets for any var in mnist.variables
mnist.sets
# ['train_set','valid_set','test_set']
# for the case of MNIST, the data comes already formatted in 3 sets

#95-129
####################
# Create own dataset
####################

# suppose we have a bunch of images, labels and an extra variable 
# s.a. noise_level, and we have given a train and test set 

train_images  = np.random.randn(2000,4,5,5)
train_labels  = np.random.randint(0,3,2000)
train_noise_l = np.random.rand(2000,2)

test_images  = np.random.randn(1000,4,5,5)
test_labels  = np.random.randint(0,3,1000)
test_noise_l = np.random.rand(1000,2)

# first initialize the dataset with some attributes (or none)
my_dataset = sk.dataset.Dataset(name='personnal',data_format='NCHW',
                datum_shape=(4,5,5),n_classes=2)

# then add the variables, the variables always are added as
# a dictionnary with key being their name used in the API
my_dataset.add_variable({'images':{'train_set':train_images,
                                    'test_set':test_images},
                        'labels':{'train_set':train_labels,
                                    'test_set':test_labels},
                        'noise_l':{'train_set':train_noise_l,
                                    'test_set':test_noise_l}})
my_dataset.variables
#['images,'labels','noise_l']
my_dataset.sets
#['train_set','test_set']
my_dataset.n_train_set
# 2000

# 131-160
###################
# Dataset splitting
###################

# Suppose wejust have some training data, we first cast it
# into a Dataset object
train_images  = np.random.randn(2000,4,5,5)
train_labels  = np.random.randint(0,3,2000)
train_noise_l = np.random.rand(2000,2)

# first initialize the dataset with some attributes (or none)
my_dataset = sk.dataset.Dataset(name='personnal',data_format='NCHW',
                datum_shape=(4,5,5),n_classes=2)

# then add the training variables
my_dataset.add_variable({'images':{'train_set':train_images},
                        'labels':{'train_set':train_labels},
                        'noise_l':{'train_set':train_noise_l}})

# then create first a test set then a valid set
my_dataset.split_set("train_set","test_set",0.25)
my_dataset.split_set("train_set","valid_set",0.2)

my_dataset.n_train_set
#1200
my_dataset.n_valid_set
#300
my_dataset.n_test_set
#500





