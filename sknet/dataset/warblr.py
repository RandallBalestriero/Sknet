import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read

class warblr:
    """`freefield1010 <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_.

    Our second dataset comes from a UK bird-sound crowdsourcing 
    research spinout called Warblr. From this initiative we have 
    10,000 ten-second smartphone audio recordings from around the UK. 
    The audio totals around 44 hours duration. The audio will be 
    published by Warblr under a Creative Commons licence. The audio 
    covers a wide distribution of UK locations and environments, and 
    includes weather noise, traffic noise, human speech and even human 
    bird imitations. It is directly representative of the data that is 
    collected from a mobile crowdsourcing initiative.
    """
    def __init__(self):
        pass
    def load(self,n_data=7000,seed=None,train_valid = [0.6,0.7]):
        self.n_data = n_data
        self.classes = 2
        self.name = 'warblr'
        t = time.time()
        PATH = os.environ['DATASET_PATH']
        if not os.path.isdir(PATH+'warblr'):
            print('Creating Directory')
            os.mkdir(PATH+'warblr')

        if not os.path.exists(PATH+'warblr/warblrb10k_public_wav.zip'):
            print('Downloading Wav Data')
            url = 'https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip'
            urllib.request.urlretrieve(url,PATH+'warblr/warblrb10k_public_wav.zip')

        if not os.path.exists(PATH+'warblr/warblrb10k_public_metadata.csv'):
            print('Downloading Wav Data')
            url = 'https://ndownloader.figshare.com/files/6035817'
            urllib.request.urlretrieve(url,PATH+'warblr/warblrb10k_public_metadata.csv')  

        print('Loading warblr')
        #Loading Labels
        labels = np.loadtxt(PATH+'warblr/warblrb10k_public_metadata.csv',
                delimiter=',',skiprows=1,dtype='str')[:n_data]
        # Loading the files
        f    = zipfile.ZipFile(PATH+'warblr/warblrb10k_public_wav.zip')
        N       = labels.shape[0]
        wavs    = list()
        for i,files_ in enumerate(labels):
            wavfile   = f.read('wav/'+files_[0]+'.wav')
            byt       = io.BytesIO(wavfile)
            wavs.append(wav_read(byt)[1].astype('float32'))
#            print(len(wav_read(byt)[1].astype('float32')))
#            wavs[i+1] = wav_read(byt)[1].astype('float32')
        labels = labels[:,1].astype('int32')
        # the test set is closed from the competition thus we
        # split the training set into train, valid and test set
        # Compute a Valid Set
        random_indices = np.random.RandomState(seed=seed).permutation(N)
        train_indices  = random_indices[:int(train_valid[0]*N)]
        valid_indices  = random_indices[int(train_valid[0]*N):int((train_valid[1])*N)]
        test_indices   = random_indices[int(train_valid[1]*N):]
        train_set      = [[wavs[i] for i in train_indices],labels[train_indices]]
        valid_set      = [[wavs[i] for i in valid_indices],labels[valid_indices]]
        test_set       = [[wavs[i] for i in test_indices],labels[test_indices]]
        self.datum_shape = (None)
            
        print('Dataset warblr loaded in','{0:.2f}'.format(time.time()-t),'s.')

        return train_set, valid_set, test_set


