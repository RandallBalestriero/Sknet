import sknet
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pylab as pl
import time
import tensorflow as tf


# Data Loading
#-------------

dataset = sknet.dataset.load_mini()
dataset.split_set("train_set","test_set",0.25)
dataset.split_set("train_set","valid_set",0.1)

# Model
#------

batch_size  = 64

input_shape = [batch_size]+list(dataset.datum_shape)



layers = [sknet.layer.Input(input_shape=input_shape,data_format=dataset.data_format)]
layers.append(sknet.layer.Dense(layers[-1],units=45))
layers.append(sknet.layer.Activation(layers[-1],0))
layers.append(sknet.layer.Dense(layers[-1],units=3))
layers.append(sknet.layer.Activation(layers[-1],0))
layers.append(sknet.layer.Dense(layers[-1],units=4))

network = sknet.network.Network(layers,name='-model(cnn.base)')



# Loss and Optimizer
#-------------------

lr_schedule = sknet.optimize.schedule.stepwise({0:0.001,50:0.0001,100:0.00005})

entropy_loss = sknet.optimize.loss.sparse_crossentropy_logits(labels=None,
        q=network[-1])
sparsity_loss = sknet.optimize.loss.l2_norm(network[1].W)
training_loss = entropy_loss+sparsity_loss

network.add_loss(training_loss,'train',sknet.optimize.optimizer.Adam(),lr_schedule)

# any of the two works
# accuracy = sknet.optimize.loss.accuracy(labels=entropy_loss.labels,prediction=network[-1])
accuracy = sknet.optimize.loss.accuracy(labels=None,predictions=network[-1])

network.add_variable(accuracy,'accuracy')


# Pipeline
#---------


linkage     = [[network[0],"images"],
                [entropy_loss.labels,"labels"],
                [accuracy.labels,"labels"]]


network.add_linkage(linkage)



pipeline = sknet.utils.Pipeline(network,dataset)




pipeline.monitor([[pipeline.network[-1].sparse_crossentropy_logits.loss,]
                    [pipeline.network[2].mask,"every 1 epoch"],
                    [pipeline.network[4].mask,"every 1 epoch"]])

# Training
#---------

pipeline.fit(n_epochs=120)



h = 200
X = dataset['images']['train_set']
Y = dataset['labels']['train_set']

x_min, x_max = X[:, 0].min() - .15, X[:, 0].max() + .15
y_min, y_max = X[:, 1].min() - .15, X[:, 1].max() + .15
xx, yy = np.meshgrid(np.linspace(x_min, x_max, h),np.linspace(y_min, y_max, h))
xxx=xx.flatten()
yyy=yy.flatten()
XX = np.asarray([xxx,yyy]).astype('float32').T

linkage     = [[pipeline.network[0],XX]]
predictions = pipeline.predict([pipeline.network[2].mask,pipeline.network[4].mask,pipeline.network[5]],linkage=linkage,concat=True)

others = pipeline.predict([pipeline.network[2],pipeline.network[4],pipeline.network[5]],linkage=linkage,concat=True)


for i in range(45):
    pl.figure(figsize=(6,6))
    pl.imshow(np.flipud(others[0][:,i].reshape((200,200))),aspect='auto',extent=[x_min,x_max,y_min,y_max])
    pl.xticks([])
    pl.yticks([])

    pl.savefig('layer1_unit'+str(i)+'.png')
    pl.close



for i in range(3):
    pl.figure(figsize=(6,6))
    pl.imshow(np.flipud(others[1][:,i].reshape((200,200))),aspect='auto',extent=[x_min,x_max,y_min,y_max])
    pl.xticks([])
    pl.yticks([])

    pl.savefig('layer2_unit'+str(i)+'.png')
    pl.close


for i in range(4):
    pl.figure(figsize=(6,6))
    pl.imshow(np.flipud(others[2][:,i].reshape((200,200))),aspect='auto',extent=[x_min,x_max,y_min,y_max])
    pl.xticks([])
    pl.yticks([])

    pl.savefig('layer3_unit'+str(i)+'.png')
    pl.close



exit()




indices = predictions[2].argmax(1)
predictions[2]*=0
predictions[2][range(predictions[0].shape[0]),indices]=1
predictions[2] = predictions[2].astype('bool')

plot1  = sknet.utils.geometry.get_input_space_partition(predictions[0],h,h)
plot2  = sknet.utils.geometry.get_input_space_partition(predictions[1],h,h)
plot3  = sknet.utils.geometry.get_input_space_partition(predictions[2],h,h)

plot12  = sknet.utils.geometry.get_input_space_partition(np.concatenate([predictions[0],predictions[1]],1),h,h)
plot123 = sknet.utils.geometry.get_input_space_partition(np.concatenate([predictions[0],
                                        predictions[1],predictions[2]],1),h,h)



pl.figure(figsize=(6,6))
pl.imshow(np.flipud(plot1),aspect='auto',cmap='Greys',extent=[x_min,x_max,y_min,y_max],alpha=0.5)
for k in range(4):
    indices = pl.where(Y==k)[0]
    print(np.shape(X))
    indices = indices[::16]
    pl.plot(X[indices,0],X[indices,1],'x',alpha=0.3)

pl.xticks([])
pl.yticks([])

#pl.show()
pl.savefig('layer1.png')
pl.close


#exit()

pl.figure(figsize=(6,6))
pl.imshow(np.flipud(plot12),aspect='auto',cmap='Greys',extent=[x_min,x_max,y_min,y_max],alpha=0.5)
for k in range(4):
    indices = pl.where(Y==k)[0]
    print(np.shape(X))
    indices = indices[::16]
    pl.plot(X[indices,0],X[indices,1],'x',alpha=0.3)

pl.xticks([])
pl.yticks([])

pl.savefig('layer12.png')
pl.close


pl.figure(figsize=(6,6))
pl.imshow(np.flipud(plot123),aspect='auto',cmap='Greys',extent=[x_min,x_max,y_min,y_max],alpha=0.5)

for k in range(4):
    indices = pl.where(Y==k)[0]
    print(np.shape(X))
    indices = indices[::16]
    pl.plot(X[indices,0],X[indices,1],'x',alpha=0.3)

pl.xticks([])
pl.yticks([])

pl.savefig('layer123.png')
pl.close









