
class Model:
    def __init__(self,input_shape, n_classes=10, nonlinearity=tf.nn.leaky_relu, batch_norm=True, K=4, L=3,trainable=True):
        self.n_classes    = n_classes
        self.input_shape  = input_shape
        self.trainable    = trainable
        self.nonlinearity = nonlinearity
        self.batch_norm   = batch_norm
	# Those variables are only used for toy examples
        self.K            = K
        self.L            = L














