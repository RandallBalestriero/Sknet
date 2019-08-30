import tensorflow as tf
from . import layers, ops
from .ops import Op
from .layers import Layer

# ToDo models : LeNet LeNet5 smallCNN largeCNN allCNN

class Network:
    def __init__(self, layers=[], name='model'):
        self.name = name
        self.layers = layers

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Network(layers=self.layers[key], name='sub'+self.name)
        return self.layers[key]

    def __len__(self):
        return len(self.layers)

    def append(self, item):
        """append an additional layer to the current network"""
        if hasattr(item, '__len__'):
            item = list(item)
            self.layers += item
        else:
            self.layers.append(item)
    def as_list(self):
        """return the layers as a list"""
        return [layer for layer in self]

    def deter_dict(self, value):
        deter_dict = dict()
        for item in self:
            if isinstance(item, Layer):
                for i in item.internal_ops:
                    if hasattr(i, 'deter_dict'):
                        deter_dict.update(i.deter_dict(value))
            else:
                if hasattr(item, 'deter_dict'):
                    deter_dict.update(item.deter_dict(value))
        return deter_dict

    @property
    def shape(self):
        """return the list of shapes of the feature maps for all the
        layers currently in the network."""
        return [i.get_shape().as_list() for i in self]

    @property
    def reset_variables_op(self, group=True):
        """gather all the reset variables op of each of the layers
        and group them into a single op if group is True, or
        return the list of operations"""
        var = []
        for layer in self:
            if hasattr(layer, 'variables'):
                var.append(layer.reset_variables_op)
        if group:
            return tf.group(*var)
        return var

    def variables(self, trainable=True):
        """return all the variables of the network
        which are trainable only or all"""
        var = list()
        for layer in self:
            if hasattr(layer, 'variables'):
                var += layer.variables(trainable)
        return var

    def backward(self, tensor):
        """feed the tensor backward in the network by
        successively calling each layer backward method,
        from the last layer to the first one. Usefull when
        doing backpropagation to get the gradient w.r.t. the input"""
        ops = self.as_list()[::-1]
        for op in ops:
            tensor = op.backward(tensor)
        return tensor

    @property
    def updates(self):
        """gather all the network updates that are
        present in the layer (for example the passive
        update of the batch-normalization layer"""
        updates = []
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates
        return updates


def Resnet(dnn, n_classes=None, D=4, W=1, block=layers.ResBlockV1, global_pool=True):
    UNITS = [16*W, 32*W, 64*W]
    dnn.append(ops.Conv2D(dnn[-1], filters=(UNITS[0], 3, 3), pad='same'))
    dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
    for bl, units in enumerate(UNITS):
        for d in range(D):
            dnn.append(block(dnn[-1], units))
        if bl < 2:
            dnn.append(block(dnn[-1], UNITS[bl+1], stride=2))
            dnn.append(ops.Pool2D(dnn[-1], (2, 2), pool_type='AVG'))
#    dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
    dnn.append(ops.Activation(dnn[-1], 0.01))
    if global_pool:
        dnn.append(ops.GlobalPool2D(dnn[-1], pool_type='AVG'))
    if n_classes is not None:
        dnn.append(ops.Dense(dnn[-1], n_classes))


def ConvLarge(dnn, n_classes=None):

    dnn.append(layers.Conv2D(dnn[-1], (96, 3, 3), pad='same',
                             nonlinearity=0.01))
    dnn.append(layers.Conv2D(dnn[-1], (96, 3, 3), pad='full',
                             nonlinearity=0.01))
    dnn.append(layers.Conv2D(dnn[-1], (96, 3, 3), pad='same',
                               nonlinearity=0.01))
    dnn.append(layers.Conv2DPool(dnn[-1], (96, 3, 3), pad='full',
                                 nonlinearity=0.01, pool_shape=(2, 2)))

    dnn.append(layers.Conv2D(dnn[-1], (192, 3, 3), pad='valid',
                             nonlinearity=0.01))
    dnn.append(layers.Conv2D(dnn[-1], (192, 3, 3), pad='full',
                             nonlinearity=0.01))
    dnn.append(layers.Conv2D(dnn[-1], (192, 3, 3), pad='same',
                               nonlinearity=0.01))
    dnn.append(layers.Conv2DPool(dnn[-1], (192, 3, 3), pad='valid',
                                 nonlinearity=0.01, pool_shape=(2, 2)))

    dnn.append(layers.Conv2D(dnn[-1], (192, 3, 3), pad='valid',
                             nonlinearity=0.01))
    dnn.append(layers.Conv2D(dnn[-1], (192, 3, 3), pad='same',
                               nonlinearity=0.01))
    dnn.append(layers.Conv2D(dnn[-1], (192, 1, 1), pad='valid',
                             nonlinearity=0.01))

    if n_classes is not None:
        dnn.append(ops.Conv2D(dnn[-1], (n_classes, 1, 1)))
        dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
        dnn.append(ops.GlobalPool2D(dnn[-1], pool_type='AVG', keep_dims=False))


def ConvSmall(dnn, n_classes=None):

    dnn.append(layers.Conv2DPool(dnn[-1], (32, 5, 5), pad='full',
                                 nonlinearity=0.01, pool_shape=(2, 2)))
    dnn.append(layers.Conv2D(dnn[-1], (64, 3, 3), nonlinearity=0.01,
                             pad='valid'))
    dnn.append(layers.Conv2DPool(dnn[-1], (64, 3, 3), pad='full',
                                 nonlinearity=0.01, pool_shape=(2, 2)))

    dnn.append(layers.Conv2D(dnn[-1], (128, 3, 3), pad='valid',
                             nonlinearity=0.01))
    if n_classes is not None:
        dnn.append(ops.Conv2D(dnn[-1], (n_classes, 1, 1), b=None))
        dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
        dnn.append(ops.GlobalPool2D(dnn[-1], pool_type='AVG', keep_dims=False))



def DenseNet(dnn, num_classes=None, k=32, theta=0.5):

    dnn.append(ops.Conv2D(dnn[-1], (2*k, 7, 7), b=None, pad='same', strides=2))
    prev_kernels = 2*k
    input_kernels = prev_kernels

    ###########
    for ITER in [6, 12, 24, 16]:
        layers_concat = list()
        for i in range(ITER):
            cur_kernels = 4 * k
            dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
            dnn.append(ops.Activation(dnn[-1], 0.))
            dnn.append(ops.Conv2D(dnn[-1], (cur_kernels, 1, 1), b=None))
#            dnn.append(ops.Dropout(dnn[-1], 0.2))

            cur_kernels = input_kernels + (k * i)
            dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
            dnn.append(ops.Activation(dnn[-1], 0.))
            dnn.append(ops.Conv2D(dnn[-1], (cur_kernels, 3, 3), b=None,
                       pad='same'))
#            dnn.append(ops.Dropout(dnn[-1], 0.2))

            layers_concat.append(dnn[-1])

        cur_layer = tf.concat(layers_concat, 1)
        print(cur_layer)
        prev_kernels = cur_kernels

        dnn.append(ops.BatchNorm(cur_layer, [0, 2, 3]))
        dnn.append(ops.Conv2D(dnn[-1], (int(prev_kernels*theta), 1, 1), b=None))
#        dnn.append(ops.Dropout(dnn[-1], 0.2))
        dnn.append(ops.Pool2D(dnn[-1], (2, 2), pool_type='AVG'))

        prev_kernels = int(prev_kernels*theta)
        input_kernels = prev_kernels
    dnn.append(ops.BatchNorm(dnn[-1], [0, 2, 3]))
    dnn.append(ops.Activation(dnn[-1], 0.01))
    dnn.append(ops.GlobalPool2D(dnn[-1], pool_type='AVG'))
    if num_classes is not None:
        dnn.append(ops.Dense(dnn[-1], num_classes))





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
