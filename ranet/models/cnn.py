from .model import Model



class DenseCNN(Model):
    def get_layers(self,input_variable,input_shape,training):
        layers = [InputLayer(input_shape,input_variable)] #(? ? 1-3)
        layers.append(FirstConv2DLayer(layers[-1],filters_T=9,sampling_n=5,filter_shape=5,spline=False)) #(? ? 32)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) # (? ? 32)
        layers.append(Pool2DLayer(layers[-1],2)) #(? ? 32)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=17,channels_T=9,filter_shape=3,spline=False)) # -> n=5 -> (? ? 64)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(Pool2DLayer(layers[-1],2))#(? ? 64)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=33,channels_T=17,filter_shape=3,spline=False)) # -> n=5 -> (? ? 128)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(GlobalPoolLayer(layers[-1]))
        layers.append(DenseLayer(layers[-1],self.n_classes))
        return layers[-1].output,layers



class DenseCNN2(Model):
    def get_layers(self,input_variable,input_shape,training):
        layers = [InputLayer(input_shape,input_variable)] #(? ? 1-3)
        layers.append(FirstConv2DLayer(layers[-1],filters_T=9,sampling_n=10,filter_shape=5,spline=False)) #(? ? 32)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) # (? ? 32)
        layers.append(Pool2DLayer(layers[-1],2)) #(? ? 32)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=17,channels_T=9,filter_shape=3,spline=False)) # -> n=5 -> (? ? 64)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(Pool2DLayer(layers[-1],2))#(? ? 64)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=33,channels_T=17,filter_shape=3,spline=False)) # -> n=5 -> (? ? 128)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(GlobalPoolLayer(layers[-1]))
        layers.append(DenseLayer(layers[-1],self.n_classes))
        return layers[-1].output,layers






############################



class largeCNN:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],64,5,pad='SAME',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(ConvLayer(layers[-1],96,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],96,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,1,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
		self.layers = layers
                return self.layers



class smallCNN:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],64,5,pad='SAME',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],128,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,1,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
		self.layers = layers
                return self.layers








class allCNN1:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],5,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (5 28 28) : 3920
                layers.append(ConvLayer(layers[-1],7,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (7 24 24) : 4032
                layers.append(ConvLayer(layers[-1],9,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (9 22 22) : 4356
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allCNN2:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],7,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (7 28 28)  : 5488
                layers.append(ConvLayer(layers[-1],11,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (11 24 24) : 6336
                layers.append(ConvLayer(layers[-1],15,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (15 22 22) : 7260
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allCNN3:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],12,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (12 28 28) : 9408
                layers.append(ConvLayer(layers[-1],16,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (16 24 24) : 9216
                layers.append(ConvLayer(layers[-1],20,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (20 22 22) : 9680
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allCNN4:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],20,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (20 28 28) : 15680
                layers.append(ConvLayer(layers[-1],32,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (32 24 24) : 18432
                layers.append(ConvLayer(layers[-1],48,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (48 20 20) : 19200
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 2048
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 256
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers












