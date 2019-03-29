import tensorflow as tf
import numpy as np
from . import Layer



class RandomCrop(Layer):
    """Random cropping of input images

    During training, a random continuous part of the image
    of shape crop_size is extracted and used as layer output.
    During testing, the center part of the image of shape
    crop_size is used as output.

    :param incoming: The input to the layer
    :type incoming: tuple of positive int or :class:`Layer` instance
    :param crop_size: Shape of the image after crop 
    :type spam: couple of positive int

    Example of use::
    
        input_layer = Input((64,3,32,32),data_format='NCHW') # Init an input layer with input shape
        crop_layer  = RandomCrop(input_layer,(28,28))        # output of this layer is (64,3,28,28)
    """

    def __init__(self, incoming, crop_size,training=None):
        """initialize the class

        :param incoming: input shape or layer
        :type incoming: shape or :class:`Layer` instance
        :param crop_size: the size of the crop to extract from 
                          the input spatial shape
        :type crop_size: couple of int
        """
        super().__init__(incoming)
        # Set attributes
        if np.isscalar(crop_size):
            self.crop_size = [crop_size,crop_size]
        else:
            assert(len(crop_size)==2)
            self.crop_size = list(crop_size)
        # Set shape after transpose as this function needs
        # data_format='NHWC'
        if self.data_format=='NCHW':
            self.images_shape = [self.in_shape[i] for i in [0,2,3,1]]
            self.out_shape = self.in_shape[:2]+self.crop_size
        else:
            self.images_shape = self.in_shape
            self.out_shape = [self.in_shape[0]]+self.crop_size+[self.in_shape[-1]]
        if np.isscalar(crop_size):
            self.crop_size = [crop_size,crop_size]
        else:
            assert(len(crop_size)==2)
            self.crop_size = list(crop_size)
        # Number of patches of crop_size shape in the input in Height and Width
        self.n_windows_H = np.int32((self.images_shape[1]-self.crop_size[0]+1)//2)
        self.n_windows_W = np.int32((self.images_shape[2]-self.crop_size[1]+1)//2)
        # Random indices
        self.indices_H = tf.cast(tf.floor(tf.random_uniform((self.images_shape[0],),
                                    maxval=np.float32(self.n_windows_H))),tf.int32)
        self.indices_W = tf.cast(tf.floor(tf.random_uniform((self.images_shape[0],),
                                    maxval=np.float32(self.n_windows_W))),tf.int32)
        # Check if can perform forward now
        if self.given_input:
            self.forward(incoming.output,training)
    def forward(self,input,training,**kwargs):
        if self.data_format=='NCHW':
            input = tf.transpose(input,[0,2,3,1])
        # Extract patches of the crop_size shape
        input_patches  = tf.extract_image_patches(input,[1]+self.crop_size+[1],
                            strides=[1,1,1,1],rates=[1,1,1,1],padding='VALID')
        random_indices = tf.stack([tf.range(self.in_shape[0]),
                                        self.indices_H,
                                        self.indices_W],1)
        random_patches = tf.gather_nd(input_patches,random_indices)
        # Now take the center parts
        center_indices = tf.stack([tf.range(self.in_shape[0]),
                            tf.fill((self.images_shape[0],),self.n_windows_H),
                            tf.fill((self.images_shape[0],),self.n_windows_W)],1)
        center_patches = tf.gather_nd(input_patches,center_indices)
        output = tf.reshape(tf.cond(training, lambda :random_patches,
                                lambda :center_patches),
                                [self.out_shape[0]]+self.crop_size+[self.images_shape[-1]])
        # Convert back to original data_format
        if self.data_format=='NCHW':
            self.output = tf.transpose(output,[0,3,1,2])
        else:
            self.output = output
        return self.output


class RandomAxisReverse(Layer):
    """randomly reverse an axis of the input
    This layer randomly reverse (or flip) one (or multiple) axis
    in its input.
    """
    def __init__(self, incoming, axis, training=None):
        """initialize the class

        :param incoming: the input shape or the incoming layer
        :type incoming: shape or :class:`Layer` instance
        :param axis: the axis to randomly flip
        :type axis: list or tuple of ints
        :param training: a boolean variable describing if the model
                         is in train or test phase
        :type training: tf.bool
        """
        super().__init__(incoming)
        prob = tf.random_uniform((self.in_shape[0],))
        self.to_reverse = tf.less(prob,0.5)
        self.out_shape = self.in_shape
        self.axis = axis
        if self.given_input:
            self.forward(incoming.output,training)
    def forward(self,input,training):
        reverse_input = tf.where(self.to_reverse,
                                tf.reverse(input,self.axis),input)
        self.output = tf.cond(training,lambda :input,lambda :reverse_input)
        return self.output


class RandomRot90(Layer):
    """ranodm rotate by 90 degrees the input
    This layer performs a random rotation of the input to 90 degrees
    this can be clockwise or counter clockwise with same probability
    """
    def __init__(self, incoming, training=None):
        """initialize the class

        :param incoming: the input shape or the incoming layer
        :type incoming: shape or instalce of :class:`Layer`
        :param training: boolean describing if the model is in
                         trianing or testing mode, should be left
                         None in most cases
        :type training: tf.bool
        """
        super().__init__(incoming)
        prob = tf.random_uniform((self.in_shape[0],),maxval=np.float32(3))
        self.rot_left = tf.less(prob,1)
        self.rot_right = tf.greater(prob,2)
        self.out_shape = self.in_shape
        if self.given_input:
            self.forward(incoming.ouput,training)
    def forward(self,input,training,**kwargs):
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
        self.output = tf.cond(training,lambda :images,lambda :incoming.output)
        return self.output





