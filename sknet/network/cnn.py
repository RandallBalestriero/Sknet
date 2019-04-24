import tensorflow as tf
from .. import layers,ops
from . import Network

#ToDo models : LeNet LeNet5 smallCNN largeCNN allCNN

def ConvLarge(dnn):

    dnn.append(layers.Conv2D(dnn[-1],[(96,3,3),{'b':None,'pad':'same'}],
                                    [[0,2,3]], [0.01]))
    dnn.append(layers.Conv2D(dnn[-1],[(96,3,3),{'b':None,'pad':'full'}],
                                    [[0,2,3]], [0.01]))
    dnn.append(layers.Conv2DPool(dnn[-1],[(96,3,3),{'b':None,'pad':'full'}],
                                    [[0,2,3]], [0.01],[(1,2,2)]))

    dnn.append(layers.Conv2D(dnn[-1],[(192,3,3),{'b':None}],
                                    [[0,2,3]], [0.01]))
    dnn.append(layers.Conv2D(dnn[-1],[(192,3,3),{'b':None,'pad':'full'}],
                                    [[0,2,3]], [0.01]))
    dnn.append(layers.Conv2DPool(dnn[-1],[(192,3,3),{'b':None}],
                                    [[0,2,3]], [0.01],[(1,2,2)]))

    dnn.append(layers.Conv2D(dnn[-1],[(192,3,3),{'b':None}],
                                    [[0,2,3]], [0.01]))
    dnn.append(layers.Conv2D(dnn[-1],[(192,1,1),{'b':None}],
                                    [[0,2,3]], [0.01]))
    dnn.append(layers.Conv2DPool(dnn[-1],[(10,1,1),{'b':None}],
                                    [[0,2,3]], [0.01],[(1,-1,-1),
                                    {'pool_type':'AVG','keep_dims':False}]))

    return dnn


def ConvSmall(dnn):

    dnn.append(layers.Conv2DPool(dnn[-1],[(32,5,5),{'b':None,'pad':'full'}],
                                    [[0,2,3]], [0.01],[(1,2,2)]))
    dnn.append(layers.Conv2D(dnn[-1],[(64,3,3),{'b':None}],
                                    [[0,2,3]], [0.01]))
    dnn.append(layers.Conv2DPool(dnn[-1],[(64,3,3),{'b':None,'pad':'full'}],
                                    [[0,2,3]], [0.01],[(1,2,2)]))

    dnn.append(layers.Conv2D(dnn[-1],[(128,3,3),{'b':None}],
                                    [[0,2,3]], [0.01]))
    dnn.append(layers.Conv2DPool(dnn[-1],[(10,1,1),{'b':None}],
                                    [[0,2,3]], [0.01],[(1,-1,-1),
                                    {'pool_type':'AVG','keep_dims':False}]))
    return dnn






def Multiscale2(dnn):
    start    = len(dnn)
    indices  = [start+i for i in [1,3,5,7,9,11]]
    my_layer = layers.custom_layer(ops.Dense,ops.BatchNorm,ops.Activation)

    dnn.append(layers.Conv2D(dnn[-1],[(128,3,3),{'b':None,'pad':'same'}],
                                    [[0,2,3]], [0.01]))
    dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

    dnn.append(layers.Conv2DPool(dnn[-2],[(256,3,3),{'b':None,'pad':'same'}],
                                    [[0,2,3]],[0.01],[(1,2,2)]))
    dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

    dnn.append(layers.Conv2D(dnn[-2],[(512,3,3),{'b':None,'pad':'same'}],
                                    [[0,2,3]], [0.01]))
    dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

    dnn.append(layers.Conv2DPool(dnn[-2],[(512,3,3),{'b':None,'pad':'same'}],
                                    [[0,2,3]],[0.01],[(1,2,2)]))
    dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

    dnn.append(layers.Conv2D(dnn[-2],[(1024,3,3),{'b':None,'pad':'same'}],
                                    [[0,2,3]], [0.01]))
    dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

    dnn.append(layers.Conv2D(dnn[-2],[(1024,3,3),{'b':None}],[[0,2,3]],
                                                [0.01]))
    dnn.append(my_layer(dnn[-1],[10,{'b':None}],[0],[tf.identity]))

    return dnn,indices
















