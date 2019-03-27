import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read

class freefield1010:
    """`freefield1010 <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_.

    freefield1010: a collection of over 7,000 excerpts from field recordings 
    around the world, gathered by the FreeSound project, and then standardised 
    for research. This collection is very diverse in location and environment, 
    and for the BAD Challenge we have newly annotated it for the 
    presence/absence of birds.
    """
    def __init__(self):
        pass
    def load(self,n_data=7000,seed=None,train_valid = [0.6,0.7]):
        self.n_data = n_data
        self.classes = 2
        self.name = 'freefield1010'
        t = time.time()
        PATH = os.environ['DATASET_PATH']
        if not os.path.isdir(PATH+'freefield1010'):
            print('Creating Directory')
            os.mkdir(PATH+'freefield1010')

        if not os.path.exists(PATH+'freefield1010/ff1010bird_wav.zip'):
            print('Downloading Wav Data')
            url = 'https://archive.org/download/ff1010bird/ff1010bird_wav.zip'
            urllib.request.urlretrieve(url,PATH+'freefield1010/ff1010bird_wav.zip')  

        if not os.path.exists(PATH+'freefield1010/ff1010bird_metadata.csv'):
            print('Downloading Wav Data')
            url = 'https://ndownloader.figshare.com/files/6035814'
            urllib.request.urlretrieve(url,PATH+'freefield1010/ff1010bird_metadata.csv')  

        print('Loading freefield1010')
        #Loading Labels
        labels = np.loadtxt(PATH+'freefield1010/ff1010bird_metadata.csv',
                delimiter=',',skiprows=1,dtype='int32')[:n_data]
        # Loading the files
        f    = zipfile.ZipFile(PATH+'freefield1010/ff1010bird_wav.zip')
        # Load the first file to get the time length (same for all files)
        wavfile = f.read('wav/'+str(labels[0,0])+'.wav')
        byt     = io.BytesIO(wavfile)
        wav     = wav_read(byt)[1]
        # Init. the wavs matrix
        N       = labels.shape[0]
        wavs    = np.empty((N,len(wav)),dtype='float32')
        wavs[0] = wav.astype('float32')
        for i,files_ in enumerate(labels[1:]):
            wavfile   = f.read('wav/'+str(files_[0])+'.wav')
            byt       = io.BytesIO(wavfile)
            wavs[i+1] = wav_read(byt)[1].astype('float32')
        labels = labels[:,1]
        # the test set is closed from the competition thus we
        # split the training set into train, valid and test set
        # Compute a Valid Set
        random_indices = np.random.RandomState(seed=seed).permutation(N)
        train_indices  = random_indices[:int(train_valid[0]*N)]
        valid_indices  = random_indices[int(train_valid[0]*N):int((train_valid[1])*N)]
        test_indices   = random_indices[int(train_valid[1]*N):]
        train_set      = [wavs[train_indices],labels[train_indices]]
        valid_set      = [wavs[valid_indices],labels[valid_indices]]
        test_set       = [wavs[test_indices],labels[test_indices]]
        self.datum_shape = (wavs.shape[1])
            
        print('Dataset ff1010bird loaded in','{0:.2f}'.format(time.time()-t),'s.')

        return train_set, valid_set, test_set


