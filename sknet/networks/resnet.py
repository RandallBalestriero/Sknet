import tensorflow as tf
from .. import layers, ops
from ..utils import upper_triangular
import numpy as np

def Resnet(dnn, n_classes=None, D=4, W=1):

    block = layers.custom_layer(ops.Conv2D, ops.BatchNorm, ops.Activation,
                                ops.Conv2D)

    #################
    for bl, units in enumerate([32*W, 64*W, 128*W]):
        dnn.append(ops.Conv2D(dnn[-1], filters=(units, 3, 3), pad='same'))
        for d in range(D):
            dnn.append(block(dnn[-1], [(units, 3, 3), {'pad': 'same'}],
                       [[0, 2, 3]], [0.01], [(units, 1, 1), {'pad': 'same'}]))
            dnn.append(ops.Merge([dnn[-1], dnn[-2]], tf.add_n))
        ##################
        if bl < 2:
            dnn.append(block(dnn[-1], [(units*2, 3, 3), {'pad': 'same'}],
                       [[0, 2, 3]], [0.01], [(units*2, 3, 3),
                                             {'pad': 'same', 'strides': 2}]))
            dnn.append(ops.Conv2D(dnn[-2], (units*2, 1, 1), strides=2))
            dnn.append(ops.Merge([dnn[-1], dnn[-2]], tf.add_n))
        elif n_classes is not None:
            dnn.append(ops.Conv2D(dnn[-1], (n_classes, 1, 1)))
            dnn.append(ops.GlobalPool2D(dnn[-1], pool_type='AVG'))

def Resnetv2(dnn, n_classes=None, D=4, W=1, shortcut='linear'):

    block = layers.custom_layer(ops.BatchNorm, ops.Activation, ops.Conv2D,
                                ops.BatchNorm, ops.Activation, ops.Conv2D)
    #################
    dnn.append(ops.Conv2D(dnn[-1], filters=(32*W, 3, 3), pad='same'))
    for bl, units in enumerate([32*W, 64*W, 128*W]):
        for d in range(D):
            dnn.append(block(dnn[-1], [[0, 2, 3]], [0.], [(units, 3, 3),
                                                          {'pad': 'same'}],
                             [[0, 2, 3]], [0.], [(units, 3, 3),
                                                 {'pad': 'same'}]))
            if shortcut == 'linear':
                dnn.append(ops.Merge([dnn[-1], dnn[-2]], tf.add_n))
            else:
                short = ops.Conv2D(dnn[-2], (units, 1, 1))
                dnn.append(ops.Merge([dnn[-1], short], tf.add_n))
        ##################
        if bl < 2:
            dnn.append(block(dnn[-1], [[0, 2, 3]], [0.], [(units, 3, 3),
                                                          {'pad': 'same',
                                                           'strides': 2}],
                             [[0, 2, 3]], [0.], [(units, 3, 3),
                                                 {'pad': 'same'}]))
            short = ops.Conv2D(dnn[-2], (units, 1, 1), strides=2)
            dnn.append(ops.Merge([dnn[-1], short], tf.add_n))
        elif n_classes is not None:
            dnn.append(ops.Conv2D(dnn[-1], (n_classes, 1, 1)))
            dnn.append(ops.GlobalPool2D(dnn[-1], pool_type='AVG'))

def OrthoResnetv2(dnn, n_classes=None, D=4, W=1, shortcut='linear', S=2):

    block = layers.custom_layer(ops.BatchNorm, ops.Activation, ops.Conv2D,
                                ops.BatchNorm, ops.Activation, ops.Conv2D)
    ortho = list()
    #################
    dnn.append(ops.Conv2D(dnn[-1], filters=(32*W, 3, 3), pad='same'))
    for bl, units in enumerate([32*W, 64*W, 128*W]):
        for d in range(D):
            dnn.append(block(dnn[-1], [[0, 2, 3]], [0.], [(units, 3, 3),
                                                          {'pad': 'same'}],
                             [[0, 2, 3]], [0.], [(units, 3, 3),
                                                 {'pad': 'same'}]))
            if shortcut != 'identity':
                dnn.append(ops.Conv2D(dnn[-2], (units, 1, 1)))
                dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
                ortho.append(tf.reshape(dnn[-2].W, (-1, units)))
                dnn.append(ops.Merge([dnn[-1], dnn[-3]], tf.add_n))
            else:
                dnn.append(ops.Merge([dnn[-1], dnn[-2]], tf.add_n))
        ##################
        if bl < 2:
            dnn.append(block(dnn[-1], [[0, 2, 3]], [0.], [(units*2, 3, 3),
                                                          {'pad': 'same',
                                                           'strides': 2}],
                             [[0, 2, 3]], [0.], [(units*2, 3, 3),
                                                 {'pad': 'same'}]))
            dnn.append(ops.Conv2D(dnn[-2], (units*2, S, S), strides=2, 
                       pad='same'))
            dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
            ortho.append(tf.reshape(dnn[-2].W, (-1, units*2)))
            dnn.append(ops.Merge([dnn[-1], dnn[-3]], tf.add_n))
        elif n_classes is not None:
            dnn.append(ops.Dense(dnn[-1], n_classes))
    return ortho

def OrthoResnetv4(dnn, n_classes, D=4):

    def block(input, bfilters, filters, strides=1):
        
        conv1 = ops.Conv2D(input, filters=(bfilters, 1, 1), b=None)
        conv = ops.Conv2D(conv1, filters=(bfilters, 3, 3), b=None, pad='same',
                          strides=strides)
        out_conv = ops.Conv2D(conv, filters=(filters, 1, 1), b=None)
        bn = ops.BatchNorm(out_conv, [0, 2, 3])
        activation = ops.Activation(bn, 0.)
        if strides > 1:
            input_shape = input.shape.as_list()
            rinput = tf.reshape(input, (input_shape[0]*input_shape[1], 1,
                                        input_shape[2], input_shape[3]))
            filters = np.array([[[1., 1.], [1., 1.]],
                                [[1., 1.], [-1., -1.]],
                                [[1., -1.], [1., -1.]],
                                [[1., -1.], [-1., 1.]]]) / np.sqrt(4)
            filters = np.expand_dims(filters.transpose((1, 2, 0,)), 2)
            convinput = ops.Conv2D(rinput, filters=(4, 2, 2), b=None,
                                   W=filters.astype('float32'),
                                   strides=strides)
            extra = [convinput]
            input = tf.reshape(convinput, (input_shape[0], input_shape[1]*4,
                                           input_shape[2]//2,
                                           input_shape[3]//2))
        else:
            extra = []
        output = ops.Merge([input, activation], tf.add_n)
        return [conv1, conv, bn, activation, out_conv]+extra+[output]

#    ortho = list()
    UNITS = [32, 128, 512]
    dnn.append(ops.Conv2D(dnn[-1], filters=(32, 3, 3), pad='same'))
    for bl, units in enumerate(UNITS):
        for d in range(D//(2**bl)):
            dnn.append(block(dnn[-1], units//(2**bl), units))
        ##################
        if bl < 2:
            dnn.append(block(dnn[-1], units, UNITS[bl+1], strides=2))
        else:
            dnn.append(ops.GlobalPool2D(dnn[-1], keepdims=False))
            dnn.append(ops.BatchNorm(dnn[-1], [0]))
            dnn.append(ops.Dense(dnn[-1], n_classes))


def OrthoResnetv3(dnn, n_classes=None, D=4, W=2, model='baseline'):

    def block(input, filters, S=1, strides=1):
        conv = ops.Conv2D(input, filters=filters, b=None, pad='same',
                          strides=strides)
        bn = ops.BatchNorm(conv, [0, 2, 3])
        activation = ops.Activation(bn, 0.)
        out_conv = ops.Conv2D(activation, filters=filters, pad='same')
        if strides > 1:
            input = ops.Conv2D(input, filters=(filters[0], S, S), b=None,
                               pad='same', strides=strides)
            extra = [input]
        else:
            extra = []
        output = ops.Merge([input, out_conv], tf.add_n)
        return [conv, bn, activation, out_conv]+extra+[output]

    ortho = list()
    S = 1 if model == 'baseline' else 3
    UNITS = [16, 32*W, 64*W, 128*W]
    #################
    dnn.append(ops.Conv2D(dnn[-1], filters=(16, 3, 3), pad='same'))
    for bl, units in enumerate(UNITS):
        for d in range(D):
            dnn.append(block(dnn[-1], (units, 3, 3)))
        ##################
        if bl < 3:
            dnn.append(block(dnn[-1], (UNITS[bl+1], 3, 3), S=S, strides=2))
            matrix = tf.reshape(dnn[-2].W, (-1, UNITS[bl+1]))
            matrix = tf.nn.l2_normalize(matrix, axis=0)
            matrix = tf.matmul(matrix, matrix, transpose_a=True)
            ortho.append(upper_triangular(matrix, strict=True))
        elif n_classes is not None:
            if model == 'baseline':
                dnn.append(ops.GlobalPool2D(dnn[-1], keepdims=False))
                dnn.append(ops.BatchNorm(dnn[-1], [0]))
            else:
                dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
            dnn.append(ops.Dense(dnn[-1], n_classes))
    return ortho
