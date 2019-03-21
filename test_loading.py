from ranet.dataset import mnist,svhn,cifar10,fashionmnist,cifar100
from pylab import *

images = list()

train_set,valid_set,test_set = mnist.load()
images.append(train_set[0][:10])

train_set,valid_set,test_set = fashionmnist.load()
images.append(train_set[0][:10])

train_set,valid_set,test_set = svhn.load()
images.append(train_set[0][:10])

train_set,valid_set,test_set = cifar10.load()
images.append(train_set[0][:10])

train_set,valid_set,test_set = cifar100.load()
images.append(train_set[0][:10])



cpt=1
for i in range(len(images)):
    for im in images[i]:
        subplot(len(images),10,cpt)
        imshow(squeeze(im.transpose([1,2,0]))/im.max(),aspect='auto')
        xticks([])
        yticks([])
        cpt+=1

savefig('test_loading.png')

