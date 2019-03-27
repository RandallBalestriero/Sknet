import urllib.request
import numpy as np
import tarfile
import os
import pickle
import time


_fine_labels = "beaver,dolphin,otter,seal,whale,\
aquarium fish,flatfish,ray,shark,trout,\
orchids,poppies,roses,sunflowers,tulips,\
bottles,bowls,cans,cups,plates,\
apples,mushrooms,oranges,pears,sweet peppers,\
clock,computer keyboard,lamp,telephone,television,\
bed,chair,couch,table,wardrobe,\
bee,beetle,butterfly,caterpillar,cockroach,\
bear,leopard,lion,tiger,wolf,\
bridge,castle,house,road,skyscraper,\
cloud,forest,mountain,plain,sea,\
camel,cattle,chimpanzee,elephant,kangaroo,\
fox,porcupine,possum,raccoon,skunk,\
crab,lobster,snail,spider,worm,\
baby,boy,girl,man,woman,\
crocodile,dinosaur,lizard,snake,turtle,\
hamster,mouse,rabbit,shrew,squirrel,\
maple,oak,palm,pine,willow,\
bicycle,bus,motorcycle,pickup truck,train,\
lawn-mower,rocket, streetcar, tank, tractor"
_coarse_labels = "aquatic mammals,fish,lowers,\
food containers,fruit and vegetables,\
household electrical devices,\
household furniture,insects,\
large carnivores,\
large man-made outdoor things,\
large natural outdoor scenes,\
large omnivores and herbivores,\
medium-sized mammals,\
non-insect invertebrates,\
people,reptiles,small mammals,\
trees,vehicles 1,vehicles 2"


class cifar100:
    """
    The `CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset is 
    just like the CIFAR-10, except it has 100 classes containing 600 images 
    each. There are 500 training images and 100 testing images per class. 
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each 
    image comes with a "fine" label (the class to which it belongs) and a 
    "coarse" label (the superclass to which it belongs).
    """
    def __init__(self,data_format='NCHW',fine_labels=True):
        self.data_format     = data_format
        self.given_test_set  = True
        self.given_valid_set = False
        self.given_unlabeled = False
        self.fine_labels     = file_labels
        if fine_labels:
            self.n_classes   = 100
            self.classes     = dict(enumerate(_fine_labels.split(',')))
        else:
            self.n_classes   = 10
            self.classes     = dict(enumerate(_coarse_labels.split(',')))
        self.name            = 'cifar100'
    def load(self):
                
        t = time.time()

        label = 'fine_labels' if self.fine_labels else 'coarse_labels'

        PATH = os.environ['DATASET_PATH']

        if not os.path.isdir(PATH+'cifar100'):
            os.mkdir(PATH+'cifar100')
            print('Creating Directory')
        if not os.path.exists(PATH+'cifar100/cifar100.tar.gz'):
            print('Downloading CIFAR100 Files')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            urllib.request.urlretrieve(url,PATH+'cifar100/cifar100.tar.gz')
        # Loading the file
        print('Loading CIFAR100')
        tar = tarfile.open(PATH+'cifar100/cifar100.tar.gz', 'r:gz')
        train_names = ['cifar-100-python/train']
        test_name   = ['cifar-100-python/test']
        for member in tar.getmembers():
            if member.name in train_names:
                f       = tar.extractfile(member)
                content = f.read()
                data_dic= pickle.loads(content,encoding='latin1')
                train_set=[data_dic['data'].reshape((-1,3,32,32)),
                            np.array(data_dic[label])]
            elif member.name in test_name:
                f       = tar.extractfile(member)
                content = f.read()
                data_dic= pickle.loads(content,encoding='latin1')
                test_set= [data_dic['data'].reshape((-1,3,32,32)),
                            np.array(data_dic[label])]
        # Check formating
        if data_format=='NHWC':
            train_set[0] = np.transpose(train_set[0],[0,2,3,1])
            test_set[0] = np.transpose(test_set[0],[0,2,3,1])
            self.datum_shape = (32,32,3)
        else:
            self.datum_shape = (3,32,32)
        print('Dataset CIFAR100 loaded in','{0:.2f}'.format(time.time()-t),'s.')

        self.train_set = train_set
        self.test_set  = test_set

