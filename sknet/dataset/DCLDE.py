import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm

from ..utils import to_one_hot, DownloadProgressBar


from . import Dataset

def load_DCLDE(subsample=1,PATH=None):
    """ToDo
"""
    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    dict_init = [('sampling_rate',44100),("n_classes",2),("path",PATH),
                ("name","freefield1010"),('classes',["no bird","bird"])]
    dataset = Dataset(**dict(dict_init))

    # Load the dataset (download if necessary) and set
    # the class attributes.

    print("Loading DCLDE")
    t = time.time()

    if not os.path.isdir(PATH+'DCLDE'):
        print('\tCreating Directory')
        os.mkdir(PATH+'DCLDE')
    if not os.path.exists(PATH+'DCLDE/DCLDE_LF_Dev.zip'):
        url = 'http://sabiod.univ-tln.fr/workspace/DCLDE2018/DCLDE_LF_Dev.zip'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                                    desc='Wav files') as t:
            urllib.request.urlretrieve(url,PATH+'DCLDE/DCLDE_LF_Dev.zip')

    # Loading the files
    f       = zipfile.ZipFile(PATH+'DCLDE/DCLDE_LF_Dev.zip')
    wavs    = list()
    labels  = list()
    for zipf in tqdm(f.filelist,ascii=True):
        if '.wav' in zipf.filename:
            wavfile   = f.read(zipf)
            byt       = io.BytesIO(wavfile)
            wavs.append(wav_read(byt)[1].astype('float32')[::subsample])
            labels.append(zipf.filename.split('/')[2])
    return wavs,labels
    dataset.add_variable({'signals':{'train_set':wavs},
                        'labels':{'train_set':labels}})

    print('Dataset freefield1010 loaded in','{0:.2f}'.format(time.time()-t),'s.')
    return dataset


