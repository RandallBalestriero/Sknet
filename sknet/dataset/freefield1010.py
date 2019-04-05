import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read

from ..utils import to_one_hot

from . import Dataset

def load_freefield1010(data_format='NCT', PATH=None):
    """Audio binary classification, presence or absence of bird songs.
    `freefield1010 <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_. 
    is a collection of over 7,000 excerpts from field recordings 
    around the world, gathered by the FreeSound project, and then standardised 
    for research. This collection is very diverse in location and environment, 
    and for the BAD Challenge we have newly annotated it for the 
    presence/absence of birds.

    :param data_format: (optional, default 'NCHW')
    :type data_format: 'NCHW' or 'NHWC'
    :param path: (optional, default $DATASET_PATH), the path to look for the data and 
                 where the data will be downloaded if not present
    :type path: str
    """
    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    if data_format=='NCT':
        datum_shape = (1,441000)
    else:
        datum_shape = (441000,1)
    dict_init = [("train_set",None),('sampling_rate',44100),
                ("datum_shape",datum_shape),("n_classes",2),
                ("n_channels",1),("spatial_shape",(441000,)),
                ("path",path),("data_format",data_format),
                ("name","freefield1010"),('classes',["no bird","bird"])]
    dataset = Dataset(**dict(dict_init))

    # Load the dataset (download if necessary) and set
    # the class attributes.
        
    print("Loading freefield1010")
    t = time.time()
        
    if not os.path.isdir(PATH+'freefield1010'):
        print('\tCreating Directory')
        os.mkdir(PATH+'freefield1010')

    if not os.path.exists(PATH+'freefield1010/ff1010bird_wav.zip'):
        print('\tDownloading freefield1010 Wav Files')
        td = time.time()
        url = 'https://archive.org/download/ff1010bird/ff1010bird_wav.zip'
        urllib.request.urlretrieve(url,PATH+'freefield1010/ff1010bird_wav.zip')  
        print("\tDone in {:.2f} s.".format(time.time()-td))

    if not os.path.exists(PATH+'freefield1010/ff1010bird_metadata.csv'):
        print('\tDownloading freefield1010 Metdata')
        td = time.time()
        url = 'https://ndownloader.figshare.com/files/6035814'
        urllib.request.urlretrieve(url,PATH+'freefield1010/ff1010bird_metadata.csv')  
        print("\tDone in {:.2f} s.".format(time.time()-td))

    #Loading Labels
    labels = np.loadtxt(PATH+'freefield1010/ff1010bird_metadata.csv',
            delimiter=',',skiprows=1,dtype='int32')
    # Loading the files
    f       = zipfile.ZipFile(PATH+'freefield1010/ff1010bird_wav.zip')
    # Load the first file to get the time length (same for all files)
    wavfile = f.read('wav/'+str(labels[0,0])+'.wav')
    byt     = io.BytesIO(wavfile)
    wav     = wav_read(byt)[1]
    # Init. the wavs matrix
    N       = labels.shape[0]
    wavs    = np.empty((N,441000),dtype='float32')
    for i,files_ in enumerate(labels[:,0]):
        wavfile   = f.read('wav/'+str(files_)+'.wav')
        byt       = io.BytesIO(wavfile)
        wavs[i]   = wav_read(byt)[1].astype('float32')
        
    labels = labels[:,1]
    wavs   = np.expand_dims(wavs,1+int(data_format=="NTC"))

    dataset.add_variable({'signals':{'train_set':wavs},
                        'labels':{'train_set':labels}})

    print('Dataset freefield1010 loaded in','{0:.2f}'.format(time.time()-t),'s.')
    return dataset


