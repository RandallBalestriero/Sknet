import urllib.request
import numpy as np
import tarfile




def load(data_format='NCHW'):

    PATH = os.environ['DATASET_PATH']

    if not os.path.isdir(PATH+'cifar100'):
        os.mkdir(PATH+'cifar100')
        print('Creating Directory')
    if not os.path.exists(PATH+'cifar100/cifar100.tar.gz'):
        print('Downloading Files')
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        urllib.request.urlretrieve(url,PATH+'cifar100/cifar100.tar.gz')
    # Loading the file
    tar = tarfile.open(PATH+'cifar100/cifar100.tar.gz', 'r:gz')
    train_names = ['cifar-100-batches-py/train']
    test_name   = ['cifar-100-batches-py/test']
    for member in tar.getmembers():
        if member.name in train_names:
            f       = tar.extractfile(member)
            content = f.read()
            lines   = tar.extractfile(UU[-1]).read()
            data_dic= pickle.loads(FF)
            train_set=[data_dic['data'].reshape((-1,3,32,32)),
                        data_dic['labels'])]
        elif member.name in test_name:
            f       = tar.extractfile(member)
            content = f.read()
            lines   = tar.extractfile(content).read()
            data_dic= pickle.loads(FF)
            test_set= [data_dic['data'].reshape((-1,3,32,32)),
                        data_dic['labels'])]
    train_set[0] = np.concatenate(train_set[0],axis=0)
    train_set[1] = np.concatenate(train_set[1],axis=0)
    # Compute a Valid Set
    random_indices = np.random.permutation(train_set[0].shape[0])
    train_indices  = random_indices[:int(train_set[0].shape[0]*0.9)]
    valid_indices  = random_indices[int(train_set[0].shape[0]*0.9):]
    valid_set    = [train_set[0][valid_indices],train_set[1][valid_indices]]
    train_set    = [train_set[0][train_indices],train_set[1][train_indices]]
    # Check formating
    if not data_format=='NCHW':
        train_set[0] = np.transpose(train_set[0],[0,2,3,1])
        test_set[0] = np.transpose(test_set[0],[0,2,3,1])
        valid_set[0] = np.transpose(valid_set[0],[0,2,3,1])
    return train_set,valid_set,test_set


