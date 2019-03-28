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


class cifar100(dict):
    """Image classification.
    The `CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset is 
    just like the CIFAR-10, except it has 100 classes containing 600 images 
    each. There are 500 training images and 100 testing images per class. 
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each 
    image comes with a "fine" label (the class to which it belongs) and a 
    "coarse" label (the superclass to which it belongs).
    """
    def __init__(self,data_format='NCHW',path=None,default_classes='fine_labels'):
        """Set up the configuration for data loading and data format

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :param path:(optional, default $DATASET_PATH), the path to look for the data and 
                    where the data will be downloaded if not present
        :type path: str
        """
        if path is None:
            path = os.environ['DATASET_PATH']
        if data_format=='NCHW':
            datum_shape = (3,32,32)
        else:
            datum_shape = (32,32,3)
        if default_classes=='fine_labels':
            classes = 100
        else:
            classes = 20
        dict_init = [("train_set",None),("test_set",None),
                    ("datum_shape",datum_shape),("n_classes",classes),
                    ("n_channels",3),("spatial_shape",(32,32)),
                    ("path",path),("data_format",data_format),("name","cifar100")]
        classes      = _fine_labels.split(',')
        superclasses = _coarse_labels.split(',')
        super().__init__(dict_init+[('classes',classes),
                    ('superclasses',superclasses)])
 
    def load(self):
        """Load the dataset (download if necessary) and adapt
        the class attributes based on the given data format.

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :return: return the train and test set, each as a couple (images,labels)
        :rtype: [(train_images,train_coarse_labels,train_fine_labels),
                (test_images,test_coarse_labels,test_fine_labels)]
        """
        print('Loading cifar100')
                
        t = time.time()

        PATH = self["path"]

        if not os.path.isdir(PATH+'cifar100'):
            print('\tCreating cifar100 Directory')
            os.mkdir(PATH+'cifar100')

        if not os.path.exists(PATH+'cifar100/cifar100.tar.gz'):
            print('\tDownloading cifar100 Dataset...')
            td = time.time()
            url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            urllib.request.urlretrieve(url,PATH+'cifar100/cifar100.tar.gz')
            print("\tDone in {:.2f}".format(time.time()-td))

        # Loading the file
        print('\tOpening cifar100')
        tar = tarfile.open(PATH+'cifar100/cifar100.tar.gz', 'r:gz')

        # Loading training set
        f         = tar.extractfile('cifar-100-python/train').read()
        data_dic  = pickle.loads(f,encoding='latin1')
        train_set = [data_dic['data'].reshape((-1,3,32,32)),
                            np.array(data_dic['coarse_labels']),
                            np.array(data_dic['fine_labels'])]

        # Loading test set
        f        = tar.extractfile('cifar-100-python/test').read()
        data_dic = pickle.loads(f,encoding='latin1')
        test_set = [data_dic['data'].reshape((-1,3,32,32)),
                            np.array(data_dic['coarse_labels']),
                            np.array(data_dic['fine_labels'])]
        # Check formating
        if self["data_format"]=='NHWC':
            train_set[0]     = np.transpose(train_set[0],[0,2,3,1])
            test_set[0]      = np.transpose(test_set[0],[0,2,3,1])

        self["train_set"] = train_set
        self["test_set"]  = test_set

        print('Dataset cifar100 loaded in','{0:.2f}'.format(time.time()-t),'s.')
