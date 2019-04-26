import tensorflow as tf
from .. import layers,ops

def Resnet(dnn,n_classes=None,D=4,W=1):

    block = layers.custom_layer(ops.BatchNorm,ops.Activation,ops.Conv2D,
                    ops.BatchNorm,ops.Activation,ops.Conv2D)

    #################
    dnn.append(block(dnn[-1],[[0,2,3]],[0.01],[(32*W,1,1)],
                [[0,2,3]],[0.01],[(32*W,3,3),{'pad':'same'}]))
    dnn.append(ops.Conv2D(dnn[-2],(32*W,1,1)))
    dnn.append(ops.Merge([dnn[-1],dnn[-2]],tf.add_n))
    for d in range(D):
        dnn.append(block(dnn[-1],[[0,2,3]],[0.01],[(32*W,1,1)],
                [[0,2,3]],[0.01],[(32*W,3,3),{'pad':'same'}]))
        dnn.append(ops.Merge([dnn[-1],dnn[-2]],tf.add_n))
    ##################
    dnn.append(block(dnn[-1],[[0,2,3]],[0.01],[(64*W,1,1)],
                [[0,2,3]],[0.01],[(64*W,3,3),{'pad':'same','strides':2}]))
    dnn.append(ops.Conv2D(dnn[-2],(64*W,1,1),strides=2))
    dnn.append(ops.Merge([dnn[-1],dnn[-2]],tf.add_n))
    for d in range(D):
        dnn.append(block(dnn[-1],[[0,2,3]],[0.01],[(64*W,1,1)],
                [[0,2,3]],[0.01],[(64*W,3,3),{'pad':'same'}]))
        dnn.append(ops.Merge([dnn[-1],dnn[-2]],tf.add_n))
    ##################
    dnn.append(block(dnn[-1],[[0,2,3]],[0.01],[(128*W,1,1)],
                [[0,2,3]],[0.01],[(128*W,3,3),{'pad':'same','strides':2}]))
    dnn.append(ops.Conv2D(dnn[-2],(128*W,1,1),strides=2))
    dnn.append(ops.Merge([dnn[-1],dnn[-2]],tf.add_n))
    for d in range(D):
        dnn.append(block(dnn[-1],[[0,2,3]],[0.01],[(128*W,1,1)],
                [[0,2,3]],[0.01],[(128*W,3,3),{'pad':'same'}]))
        dnn.append(ops.Merge([dnn[-1],dnn[-2]],tf.add_n))
    #################
    if n_classes is not None:
        dnn.append(ops.Conv2D(dnn[-1],(n_classes,1,1)))
        dnn.append(ops.GlobalPool2D(dnn[-1]))
    return dnn



