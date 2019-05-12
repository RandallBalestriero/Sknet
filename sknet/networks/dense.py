from .model import Model



class MiniDense(Model):
    def get_layers(self,input_variable,input_shape,training):
        layers = [InputLayer(input_shape,input_variable)]
        for l in range(self.L):
            layers.append(DenseLayer(layers[-1],self.K,training=training, 
                batch_norm=self.batch_norm,nonlinearity=self.nonlinearity,trainable=self.trainable))
        layers.append(DenseLayer(layers[-1],self.n_classes,training=training,batch_norm=False,nonlinearity=tf.identity))
        return layers[-1].output,layers






class smallDENSE:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],1024,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],1024,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class largeDENSE:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],4096,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],4096,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers





class allDENSE1:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],3920,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],4032,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],4356,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allDENSE2:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],5488,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],6336,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],7260,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allDENSE3:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],9408,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],9216,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],9680,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allDENSE4:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],15680,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],18432,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],19200,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers






class SpecialDense:
        def __init__(self,n_classes=10,constraint='dt',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.constraint  = constraint
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(SpecialDenseLayer(layers[-1],16,constraint=self.constraint,training=training,first=True))
#                layers.append(SpecialDenseLayer(layers[-1],64,constraint=self.constraint,training=training,first=False))
#                layers.append(DenseLayer(layers[-1],6,nonlinearity='relu',training=training))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=None,training=training))
		self.layers = layers
                return self.layers












