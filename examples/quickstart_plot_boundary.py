import sys
sys.path.insert(0, "../")

import sknet
from sknet.optimize import Adam
from sknet.optimize.loss import *
from sknet.optimize import schedule
import matplotlib
matplotlib.use('Agg')
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pylab as pl
import time
import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops,layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def smooth(image,K):
    new_image = image.astype('float32')
    for i in range(K):
        new_image+=np.roll(image,1+i,0)
        new_image+=np.roll(image,1+i,1)
    # remove boundary effects
    new_image[:K,:K]*=0.
    new_image[:K,-K:]*=0.
    new_image[-K:,:K]*=0.
    new_image[-K:,-K:]*=0.
    return (new_image>0).astype('float32')

# Data Loading
#-------------
N = 200
TIME = np.linspace(-2,2,N)
X = np.meshgrid(TIME,TIME)
X = np.stack([X[0].flatten(),X[1].flatten()],1).astype('float32')

dataset = sknet.dataset.Dataset()
dataset.add_variable({'input':{'train_set':X}})

dataset.create_placeholders(batch_size=50,
       iterators_dict={'train_set':BatchIterator("continuous")},device="/cpu:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape

dnn       = sknet.network.Network(name='simple_model')


np.random.seed(100)

b1 = (np.random.randn(6)/6).astype('float32')
W1 = (np.random.randn(2,6)/4).astype('float32')
dnn.append(ops.Dense(dataset.input, 6, W=W1, b=b1))
dnn.append(ops.Activation(dnn[-1],0.01))

b2 = (np.random.randn(6)/6).astype('float32')
W2 = (np.random.randn(6,6)/2).astype('float32')
dnn.append(ops.Dense(dnn[-1], 6, W=W2, b=b2))
dnn.append(ops.Activation(dnn[-1],0.01))

np.random.seed(int(sys.argv[-1]))
b3 = (0.101+0*np.random.randn(1)/10).astype('float32')
W3 = (np.asarray([[-0.11],[0.1],[-0.5],[-0.27],[-0.3],[-0.3]])).astype('float32')
dnn.append(ops.Dense(dnn[-1], 1 , W=W3, b=b3))

output1 = dnn[0]
output2 = dnn[2]
output3 = dnn[4]

# Workers
#---------

output = tf.concat([output1,output2,output3],-1)
output = sknet.Worker(op_name='poly',context='train_set',op= output,
        instruction='save every batch', deterministic=False)

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_worker(output)

fig = plt.figure(figsize=(6,9))
#
mask1   = output.data[0][:,:6]>0
boundary1 = sknet.utils.geometry.get_input_space_partition(mask1,N,N,1).astype('bool')
boundary10 = sknet.utils.geometry.get_input_space_partition(mask1[:,2],N,N,1).astype('bool')

mask2   = output.data[0][:,6:12]>0
boundary2 = sknet.utils.geometry.get_input_space_partition(mask2,N,N,1).astype('bool')
boundary20 = sknet.utils.geometry.get_input_space_partition(mask2[:,2],N,N,1).astype('bool')
boundary12=boundary1+boundary2

mask3   = output.data[0][:,-1]>0
boundary3 = sknet.utils.geometry.get_input_space_partition(mask3,N,N,1).astype('bool')
boundary123=boundary12+boundary3


boundary1   = boundary1.astype('float32')
boundary10  = boundary10.astype('float32')
boundary2   = boundary2.astype('float32')
boundary20  = boundary20.astype('float32')
boundary3   = boundary3.astype('float32')
boundary12  = boundary12.astype('float32')
boundary123 = boundary123.astype('float32')




#
poly1 = np.prod(output.data[0][:,:6],1)
poly12 = np.prod(output.data[0][:,:12],1)
poly123 = np.prod(output.data[0],1)

def plotit(poly,previous,b1,b2,name,last=False):
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111,projection='3d')
    ax.plot_trisurf(X[:,0], X[:,1], np.abs(poly)**0.2,
                    linewidth=0.2, antialiased=True)
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(1)
    ax.dist=9
    plt.savefig(name+'1.png', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111,projection='3d')
    if not last:
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        previous+0.7*b1,3, zdir='z', offset=0.4,cmap='Greys',
                        vmin=0,vmax=(previous+0.7*b1).max())
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        b2,3, zdir='z', offset=0.01,cmap='Greys',
                        vmin=0,vmax=b2.max())
    else:
        colors1 = plt.cm.Greys(np.linspace(0, 1, 3))
        # Red colormap which takes values from 
        colors2 = plt.cm.hsv(np.linspace(0, 1, 100))
        colors  = np.vstack((colors1[:2], colors2[[-1]]))
        # generating a smoothly-varying LinearSegmentedColormap
        cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        previous,3, zdir='z', offset=0.4,cmap='Greys',
                        vmin=0,vmax=previous.max())
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        b1,3, zdir='z', offset=0.4,cmap=cmap,
                        vmin=0,vmax=b1.max(),alpha=0.5)
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        b2,3, zdir='z', offset=0.01,cmap=cmap,
                        vmin=0,vmax=b2.max())
    WHERE  = np.where(b2>0)
    A0  = np.where(WHERE[0]<=2)[0]
    B0  = np.where(WHERE[0]>=(N-2))[0]
    A1  = np.where(WHERE[1]<=2)[0]
    B1  = np.where(WHERE[1]>=(N-2))[0]
    print(WHERE[0].min(),WHERE[0].max(),WHERE[1].min(),WHERE[1].max())
    POINTS = list()
    if len(A0)>0:
        POINTS.append(A0[0])
    if len(A1)>0:
        POINTS.append(A1[0])
    if len(B0)>0:
        POINTS.append(B0[0])
    if len(B1)>0:
        POINTS.append(B1[0])
    print(POINTS)
    for i in POINTS:

        X1 = TIME[WHERE[0][i]]
        Y1 = TIME[WHERE[1][i]]
        ax.plot([Y1,Y1],[X1,X1],[0.01,0.4],color='k',linestyle='--',zorder=100)

    ax.set_zlim((-0.08,0.47))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.dist=7
    plt.savefig(name+'2.png', bbox_inches='tight')
    plt.close()

    #


plotit(poly1,0.,boundary1,boundary10,'layer1')
plotit(poly12,boundary12,boundary2,boundary20,'layer2')
plotit(poly123,boundary123,boundary3,boundary3,'layer3',last=True)




