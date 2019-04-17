import tensorflow as tf
import numpy as np
from . import Op



class Dropout(Op):
    """Randomly mask values of the input

    This layer applies a multiplicative perturbation
    on its input by means of a binary mask. Each value
    of the mask is sampled from a Bernoulli distribution
    :math:`\mathcal{B}ernoulli(p)` where :math:`p` is the
    probability to have a :math:`1`.

    Parameters
    ----------

    incoming : :class:`Op` or shape
        the incoming layer or input shape

    p : scalar :math:`0\leq p \leq 1`
        the probability to keep the input values

    """
    def __init__(self, incoming, p=0.5, deterministic=None, **kwargs):
        super().__init__(incoming, **kwargs)

        # Set attributes
        assert(np.isscalar(p))
        self.p = p
        if self.given_input:
            self.forward(incoming.output,deterministic=deterministic)
    def forward(self,input=None, deterministic=None, _input=None, **kwargs):
        # Random indices
        if input is None:
            input = self.incoming.forward(deterministic=deterministic,_input=_input)
        random_values = tf.random_uniform(self.input_shape)
        mask          = tf.cast(tf.less(random_values,self.p),input.dtype)
        self.output = tf.cond(deterministic,lambda :input,lambda :mask*input)
        return self.output


class UniformNoise(Op):
    """Applies an additive or multiplicative Uniform noise to the input

    This layer applies an additive or multiplicative perturbation
    on its input by means of a Gaussian random variable. Each value
    of the mask is sampled from a Normal distribution
    :math:`\mathcal{U}(lower,upper)` where :math:`lower` and
    :math:`upper` are the bounds of the uniform distirbution
    Those parameters can be constant, per dimension, per channels, ... but
    there shape must be broadcastable to match the input shape.

    Parameters
    ----------

    incoming : :class:`Op` or shape
        the incoming layer or input shape

    noise_type : str, :py:data:`"additive"` or :py:data:`"multiplicative"`
        the type of noise to apply on the input

    lower : Tensor or Array
        the lower bound of the Uniform distribution

    upper : Tensor or Array
        the upper bound of the Uniform distribution

    """

    name = 'UniformNoise'
    deterministic_behavior = True

    def __init__(self, incoming, noise_type="additive", lower = np.float32(0),
            upper = np.float32(1), deterministic=None, **kwargs):
        super().__init__(incoming, **kwargs)

        # Set attributes
        self.lower = lower
        self.upper = upper
        self.noise_type = noise_type
        if self.given_input:
            self.forward(incoming.output,deterministic=deterministic)
    def forward(self,input=None, deterministic=None, _input=None, **kwargs):
        # Random indices
        random_values = tf.random_uniform(self.input_shape)\
                                *(self.upper-self.lower)-self.lower
        if input is None:
            input = self.incoming.forward(deterministic=deterministic,_input=_input)
        if self.noise_type=="additive":
            output = input+random_values
        elif self.noise_type=="multiplicative":
            output = input*random_values
        else:
            print('error')
            exit()
        self.output = tf.cond(deterministic,lambda :input,lambda :output)
        return self.output


class GaussianNoise(Op):
    """Applies an additive or multiplicative Gaussian noise to the input

    This layer applies an additive or multiplicative perturbation
    on its input by means of a Gaussian random variable. Each value
    of the mask is sampled from a Normal distribution
    :math:`\mathcal{N}(\mu,\sigma)` where :math:`\mu` is the
    mean and :math:`\sigma` the standard deviation. Those
    parameters can be constant, per dimension, per channels, ... but
    there shape must be broadcastable to match the input shape.

    Parameters
    ----------

    incoming : :class:`Op` or shape
        the incoming layer or input shape

    noise_type : str, :py:data:`"additive"` or :py:data:`"multiplicative"`
        the type of noise to apply on the input

    mu : Tensor or Array
        the mean of the Gaussian distribution

    sigma : Tensor or Array
        the standard deviation of the Gaussian distribution

    """

    name = 'GaussianNoise'
    deterministic_behavior = True

    def __init__(self, incoming, noise_type="additive", mu = np.float32(0),
            sigma = np.float32(1), deterministic=None, **kwargs):
        """init
        """
        super().__init__(incoming, **kwargs)

        # Set attributes
        self.mu         = mu
        self.sigma      = sigma
        self.noise_type = noise_type
        if self.given_input:
            self.forward(incoming.output,deterministic=deterministic)
    def forward(self,input=None, deterministic=None, _input=None, **kwargs):
        # Random indices
        random_values = tf.random_normal(self.input_shape)*self.sigma+self.mu
        if input is None:
            input = self.incoming.forward(deterministic=deterministic,_input=_input)
        if self.noise_type=="additive":
            output = input+random_values
        elif self.noise_type=="multiplicative":
            output = input*random_values
        else:
            print('error')
            exit()
        self.output = tf.cond(deterministic,lambda :input,lambda :output)
        return self.output



class RandomCrop(Op):
    """Random cropping of input images.

    During deterministic, a random continuous part of the image
    of shape crop_shape is extracted and used as layer output.
    During testing, the center part of the image of shape
    crop_shape is used as output. Apply the same perturbation
    to all the channels of the input.

    Example of use::

        input_shape = (64,3,32,32)
        # Init an input layer with input shape
        crop_layer  = RandomCrop(input_shape,(28,28))
        # output of this layer is (64,3,28,28)
        crop_layer.output_shape


    Parameters
    ----------

    incoming : :class:`Op` or shape
        the incoming layer or input shape

    crop_shape : int or couple of int
        the shape of part of the input to be
        cropped

    """

    name = 'RandomCrop'
    deterministic_behavior = True

    def __init__(self, input, crop_shape, deterministic=None, **kwargs):
        """init
        """
        # Set attributes
        if np.isscalar(crop_shape):
            self.crop_shape = [crop_shape,crop_shape]
        else:
            assert(len(crop_shape)==2)
            self.crop_shape = list(crop_shape)

        # Set shape after transpose as this function needs
        # data_format='NHWC' always
        self.spatial_shape = input.get_shape().as_list()[2:]

        # Number of patches of crop_shape shape in the input in H and W
        self.n_H = np.int64(self.spatial_shape[0]-self.crop_shape[0]+1)
        self.n_W = np.int64(self.spatial_shape[1]-self.crop_shape[1]+1)

        super().__init__(input, **kwargs)

    def forward(self,input, deterministic, **kwargs):

        # Random indices
        N,C,H,W = input.get_shape().as_list()
        random_H = tf.random_uniform((N,),maxval=np.float32(self.n_H))
        random_W = tf.random_uniform((N,),maxval=np.float32(self.n_W))
        indices_H = tf.cast(tf.floor(random_H), tf.int32)
        indices_W = tf.cast(tf.floor(random_W), tf.int32)
        random_indices = tf.stack([tf.range(N), indices_H, indices_W],1)
        # Deterministic (center) indices
        center_indices = tf.stack([tf.range(N,dtype=tf.int64), 
                        tf.fill((N,),self.n_H//2),
                        tf.fill((N,),self.n_W//2)],1)

        # need to transpose to extract patches
        input_t = tf.transpose(input,[0,2,3,1])

        # Extract patches of the crop_shape shape
        input_patches  = tf.extract_image_patches(input_t,
                                    [1]+self.crop_shape+[1],strides=[1,1,1,1],
                                    rates=[1,1,1,1],padding='VALID')
        random_patches = tf.gather_nd(input_patches,random_indices)
        center_patches = tf.gather_nd(input_patches,center_indices)

        # Set the patches to use depending if it is train or test time
        patches = tf.cond(deterministic, lambda :center_patches,
                                    lambda :random_patches)
        output  = tf.reshape(patches, [N]+self.crop_shape+[C])

        return tf.transpose(output,[0,3,1,2])


class RandomAxisReverse(Op):
    """randomly reverse axis of the input

    This layer randomly reverse (or flip) one (or multiple) axis
    in its input. It will either apply all the axis or none.
    Apply the same perturbation to all the channels of the input

    Example of use::

        # Set the input shape
        input_shape = [10,1,32,32]
        # Create the layer
        layer = RandomAxisReverse(input_shape,[2],data_format='NCHW')
        # the output will randomly put the images upside down
        # Create another case
        layer = RandomAxisReverse(input_shape,[2,3],data_format='NCHW')
        # in this case the images will randomly have both spatial
        # axis reversed

    Parameters
    ----------
    incoming : Op or shape
        the incoming layer of the input shape

    axis : int or list of int
        the axis to randomly reverse the order on

    deterministic : bool or tf.bool
        the state of the model, can be omited if the layer is not computing
        its output with the constructor
    """
    def __init__(self, incoming, axis, deterministic=None, **kwargs):
        super().__init__(incoming, **kwargs)
        self.output_shape = self.input_shape
        if np.isscalar(axis):
            self.axis = [axis]
        else:
            self.axis = axis

        if self.given_input:
            self.forward(incoming.output,deterministic=deterministic)
    def forward(self,input=None,deterministic=None,_input=None, **kwargs):
        prob = tf.random_uniform((self.input_shape[0],))
        self.to_reverse = tf.less(prob,0.5)
        if input is None:
            input = self.incoming.forward(deterministic=deterministic,_input=_input)
        reverse_input = tf.where(self.to_reverse,
                                tf.reverse(input,self.axis),input)
        self.output = tf.cond(deterministic,lambda :input,lambda :reverse_input)
        return self.output


class RandomRot90(Op):
    """randomly rotate by 90 degrees the input

    This layer performs a random rotation of the input to 90 degrees
    this can be clockwise or counter clockwise with same probability.

    :param incoming: the input shape or the incoming layer
    :type incoming: shape or instalce of :class:`Op`
    :param deterministic: boolean describing if the model is in
                     trianing or testing mode, should be left
                     None in most cases
    :type deterministic: tf.bool
    """
    def __init__(self, incoming, deterministic=None, **kwargs):
        super().__init__(incoming, **kwargs)
        self.output_shape = self.input_shape
        if self.given_input:
            self.forward(incoming.output,deterministic=deterministic)
    def forward(self,input=None,deterministic=None,_input=None, **kwargs):
        prob = tf.random_uniform((self.input_shape[0],),maxval=np.float32(3))
        self.rot_left = tf.less(prob,1)
        self.rot_right = tf.greater(prob,2)
        if input is None:
            input = self.incoming.forward(deterministic=deterministic,_input=_input)
        if self.data_format=='NCHW':
            left_rot    = tf.transpose(input,[0,1,3,2])
            left_images = tf.where(self.rot_left,left_rot,input)
            output      = tf.where(self.rot_right,tf.reverse(left_rot,[-1]),
                                                        left_images)
        else:
            left_rot    = tf.transpose(incoming.output,[0,2,1,3])
            left_images = tf.where(self.rot_left,left_rot,input)
            output      = tf.where(self.rot_right,tf.reverse(left_rot,[2]),
                                                        left_images)
        self.output = tf.cond(deterministic,lambda :input,lambda :output)
        return self.output





