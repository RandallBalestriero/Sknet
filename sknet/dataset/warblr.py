import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read

from . import Dataset

from ..utils import to_one_hot

def load_warblr(data_format='NCT', PATH=None):
    """Binary audio classification, presence or absence of a bird.
    `Warblr <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_ 
    comes from a UK bird-sound crowdsourcing 
    research spinout called Warblr. From this initiative we have 
    10,000 ten-second smartphone audio recordings from around the UK. 
    The audio totals around 44 hours duration. The audio will be 
    published by Warblr under a Creative Commons licence. The audio 
    covers a wide distribution of UK locations and environments, and 
    includes weather noise, traffic noise, human speech and even human 
    bird imitations. It is directly representative of the data that is 
    collected from a mobile crowdsourcing initiative.

    :param data_format: (optional, default 'NCHW')
    :type data_format: 'NCHW' or 'NHWC'
    :param path: (optional, default $DATASET_PATH), the path to look for the data and 
                     where the data will be downloaded if not present
    :type path: str
    """
    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    if data_format=='NCT':
        datum_shape = (1,None)
    else:
        datum_shape = (None,1)
    dict_init = [("train_set",None),('sampling_rate',44100),
                ("datum_shape",datum_shape),("n_classes",2),
                ("n_channels",1),("spatial_shape",(None,)),
                ("path",PATH),("data_format",data_format),("name","warblr"),
                ('classes',["no bird","bird"])]

    dataset = Dataset(**dict(dict_init))
        
    # Load the dataset (download if necessary) and set
    # the class attributes.
        
    print('Loading warblr')
    t = time.time()
    if not os.path.isdir(PATH+'warblr'):
        print('\tCreating Directory')
        os.mkdir(PATH+'warblr')

    if not os.path.exists(PATH+'warblr/warblrb10k_public_wav.zip'):
        print('\tDownloading Wav Files')
        td  = time.time()
        url = 'https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip'
        urllib.request.urlretrieve(url,PATH+'warblr/warblrb10k_public_wav.zip')
        print("\tDone in {:.2f} s.".format(time.time()-td))
        
    if not os.path.exists(PATH+'warblr/warblrb10k_public_metadata.csv'):
        print('\tDownloading Metadata')
        td  = time.time()
        url = 'https://ndownloader.figshare.com/files/6035817'
        urllib.request.urlretrieve(url,PATH+'warblr/warblrb10k_public_metadata.csv')  
        print("\tDone in {:.2f} s.".format(time.time()-td))

    #Loading Labels
    labels = np.loadtxt(PATH+'warblr/warblrb10k_public_metadata.csv',
            delimiter=',',skiprows=1,dtype='str')
    # Loading the files
    f    = zipfile.ZipFile(PATH+'warblr/warblrb10k_public_wav.zip')
    N    = labels.shape[0]
    wavs = list()
    exp_dim_opt = int(data_format=='NTC')
    for i,files_ in enumerate(labels):
        wavfile   = f.read('wav/'+files_[0]+'.wav')
        byt       = io.BytesIO(wavfile)
        wavs.append(np.expand_dims(wav_read(byt)[1].astype('float32'),
                                    exp_dim_opt))
    labels    = labels[:,1].astype('int32')
    dataset.add_variable({'signals':{'train_set':wavs},
                        'labels':{'train_set':labels}})

    print('Dataset warblr loaded in','{0:.2f}'.format(time.time()-t),'s.')
    return dataset

