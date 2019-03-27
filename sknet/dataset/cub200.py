import os
from tqdm import tqdm
import urllib.request
import numpy as np
import time
import io
import imageio
import tarfile
from scipy.ndimage import imread

class cub200:
    """`urbansound8k <https://urbansounddataset.weebly.com/urbansound8k.html>`_.

    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds 
    from 10 classes: air_conditioner, car_horn, children_playing, 
    dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, 
    and street_music. The classes are drawn from the 
    `urban sound taxonomy <https://urbansounddataset.weebly.com/taxonomy.html>`_. 
    The dataset is obtained from `Kaggle <https://www.kaggle.com/pavansanagapati/urban-sound-classification>_`
    """
    def __init__(self,target='class',data_format='NCHW'):
        """set up the options
        target = {class,bounding_box}
        crop-images True or False (whether to use the cropped images)
        """
        self.data_format = data_format
        self.target      = target
    def load(self,seed=None,train_valid = [0.6,0.7],):
        self.classes = 200
        self.name = 'cub200'
        t = time.time()
        PATH = os.environ['DATASET_PATH']
        if not os.path.isdir(PATH+'caltechbird'):
            print('Creating Directory')
            os.mkdir(PATH+'caltechbird')

        if not os.path.exists(PATH+'caltechbird/CUB_200_2011.tgz'):
            td = time.time()
            print('\tDownloading Data')
            url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
            urllib.request.urlretrieve(url,PATH+'caltechbird/CUB_200_2011.tgz')
            print('\tDone in {:.2f}'.format(time.time()-td))

        print('Loading caltechbird')
        tar = tarfile.open(PATH+'caltechbird/CUB_200_2011.tgz', 'r:gz')
        # Load the class names
        f     = tar.extractfile("CUB_200_2011/classes.txt")
        names = np.loadtxt(f,dtype='str')
        self.classes_names = dict()
        for c in range(self.classes):
            self.classes_names[c] = names[c,1].split('.')[1]
        # Load Bounding boxes
        f     = tar.extractfile("CUB_200_2011/bounding_boxes.txt")
        boxes = np.loadtxt(f,dtype='int32')
        bounding_boxes = dict()
        for i in range(boxes.shape[0]):
            bounding_boxes[str(boxes[i,0])]=boxes[i,1:]
        # Load dataset
        labels  = list()
        data    = [[] for i in range(200)]
        for member in tqdm(tar.getmembers()):
            if 'CUB_200_2011/images/' in member.name and 'jpg' in member.name:
                class_=member.name.split('/')[2].split('.')[0]
                image_id = member.name.split('_')[-1][:-4]
                if self.target=='class':
                    labels.append(int(class_))
                else:
                    labels.append(bounding_boxes[int(image_id)])
                f      = tar.extractfile(member)
                if self.data_format=='NCHW':
                    im = imageio.imread(f,format='jpg')
                    if len(im.shape)==2:
                        labels.pop(-1)
                        print('i')
                        continue
                    data[int(class_)].append(np.transpose(im,[2,0,1]))
                else:
                    im = imageio.imread(f,format='jpg')
                    if len(im.shape)==2:
                        labels.pop(-1)
                        print('i')
                        continue
                    data[int(class_)].append(im)
        labels = np.array(labels).astype('int32')
        # the test set is closed from the competition thus we
        # split the training set into train, valid and test set
        # Compute a Valid Set
        N = len(data)
        random_indices = np.random.RandomState(seed=seed).permutation(N)
        train_indices  = random_indices[:int(train_valid[0]*N)]
        valid_indices  = random_indices[int(train_valid[0]*N):int((train_valid[1])*N)]
        test_indices   = random_indices[int(train_valid[1]*N):]
        train_set      = [[data[i] for i in train_indices],labels[train_indices]]
        valid_set      = [[data[i] for i in valid_indices],labels[valid_indices]]
        test_set       = [[data[i] for i in test_indices],labels[test_indices]]
        if self.data_format=='NCHW':
            self.datum_shape = (3,None,None)
        else:
            self.datum_shape = (None,None,3)
            
        print('Dataset cub200 loaded in','{0:.2f}'.format(time.time()-t),'s.')

#        return train_set, valid_set, test_set


