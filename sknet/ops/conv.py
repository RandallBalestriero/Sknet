import tensorflow as tf
import tensorflow.contrib.layers as tfl
from .normalize import BatchNorm as bn
import numpy as np
from . import Op
    
from .. import utils
from .. import Variable

class Conv2D(Op):
    """2D (spatial) convolutional layer.
    Op to perform a 2D convolution onto a 4D input tensor

    :param incoming: input shape of incoming layer
    :type incoming: Op or tuple of int
    :param filters: the shape of the filters in the form 
                    (#filters, height, width)
    :type filters: triple of int
    :param nonlinearity_c: coefficient of the nonlinearity,
                           0 for ReLU,-1 for absolute value,...
    :type nonlinearity_c: scalar

    """
    name = 'Conv2D'
    deterministic_behavior = False
    def __init__(self,incoming,filters,W = tfl.xavier_initializer(),
                    b = tf.zeros, strides=1, pad='valid',
                    mode='CONSTANT', name='', W_func = tf.identity,
                    b_func = tf.identity,*args,**kwargs):
        with tf.variable_scope("Conv2DOp") as scope:
            self.scope_name = scope.original_name_scope
            self.mode       = mode
            self.strides    = strides
            self.pad        = pad
            if np.isscalar(strides):
                self.strides = [1,1,strides,strides]
            else:
                self.strides = [1,1]+list(strides)

            # Define the padding function
            if pad=='valid' or (filters[1]==1 and filters[2]==1):
                self.to_pad=False
            else:
                if pad=='same':
                    assert(filters[1]%2==1 and filters[2]%2==1)
                    self.p = [(filters[1]-1)//2,(filters[2]-1)//2]
                else:
                    self.p = [filters[1]-1,filters[2]-1]
                self.to_pad = True

            # Compute shape of the W filter parameter
            w_shape = (filters[1],filters[2],
                                incoming.shape.as_list()[1],filters[0])
            # Initialize W
            if callable(W):
                self._W = Variable(W(w_shape), name='conv2dlayer_W_'+name)
            else:
                self.W  = W
            self.W  = W_func(self._W)
            self.add_param(self._W)

            # Initialize b
            if b is None:
                self._b  = None
            elif callable(b):
                self._b = Variable(b((filters[0],1,1)),
                                        name='conv2dlayer_b_'+name)
            else:
                self._b = b
            self.b  = b_func(self._b) if b is not None else self._b
            self.add_param(self._b)

            super().__init__(incoming)

    def forward(self,input, *args,**kwargs):
        if self.to_pad:
            padded = tf.pad(input,[[0,0],[0,0],[self.p[0]]*2,
                                [self.p[1]]*2],mode=self.mode)
        else:
            padded = input
        Wx = tf.nn.conv2d(padded, self.W, strides=self.strides,
                padding='VALID', data_format="NCHW")
        if self.b is None:
            return Wx
        else:
            return Wx+self.b
    def backward(self,input):
        return tf.nn.conv2d_backprop_input(self.input.get_shape().as_list(),
                filter = self.W, out_backprop = input, strides=self.strides,
                data_format='NCHW',padding='VALID')




#######################################
#
#       1D spline
#




class SplineWaveletTransform(Op):
    """Learnable scattering network layer.

    Parameters
    ----------

    J : int
        The number of octave (from Nyquist) to decompose.

    Q : int
        The resolution (number of wavelets per octave).

    K : int
        The number of knots to use for the spline approximation of the
        mother wavelet. Should be odd, if not, will be rounded to the lesser
        odd value.

    strides : int (default 1)
        The stride for the 1D convolution.

    init : str (default gabor)
        The initialization  for the spline wavelet, can be :data:`"gabor"`,
        :data:`"random"`, :data:`"paul"`.

    trainable_scales : bool (default True)
        If the scales (dilation of the mother wavelet) should be learned

    trainable_knots : bool (default True)
        If the knots (position for the spline region boundaries) should be
        learned
    """
    deterministic_behavior = False
    name = 'SplineWaveletTransform'

    def __init__(self, input, J, Q, K, strides=1, init='random',
                 trainable_scales=False, trainable_knots=False,
                 trainable_filters=False, hilbert=False, m=None,
                 p=None, padding='valid',n_conv=None, **kwargs):
        with tf.variable_scope("SplineWaveletTransformOp") as scope:
            # Attribution
            if n_conv is None:
                n_conv = J
            K                    += (K%2)-1
            self.J,self.Q,self.K  = J, Q, K
            self.trainable_scales = trainable_scales
            self.trainable_knots  = trainable_knots
            self.trainable_filters = trainable_filters
            self.hilbert          = hilbert
            self.strides          = strides

            # ------ SCALES
            # we start with scale 1 for the nyquist and then increase 
            # the scale for the lower frequencies. This is built by using
            # a standard dyadic scale and then adding a (learnable) vector 
            # to it. As such, regularizing this delta vector constrains the 
            # learned scales to not be away from standard scales, we then 
            # sort them to have an interpretable time/frequency plot and to 
            # have coherency in case this is followed by 2D conv.
            scales = 2**(tf.range(self.J,delta=1./self.Q,dtype=tf.float32))
            delta_scales = Variable(tf.zeros(self.J*self.Q),
                                trainable=self.trainable_scales, name='scales')
            self.scales  = tf.contrib.framework.sort(scales+delta_scales)
            self._scales = np.arange(0,J,1./Q)

            # We initialize the knots  with uniform spacing 
            start = (self.K//2)
            grid  = tf.lin_space(np.float32(-start),np.float32(start), self.K)
            self.knots = Variable(grid, self.trainable_knots, name='knots')
            self.all_knots = tf.einsum('i,j->ij',self.scales,self.knots)

            # initialize m and p
            self.init_mp(m,p,init)

            # create the n_conv filter-bank(s)
            self.W  = [self.init_filters(i*J*Q//n_conv,(i+1)*J*Q//n_conv)
                                    for i in range(n_conv)]

            super().__init__(input)

    def forward(self,input,*args,**kwargs):
        # Input shape
        input_shape = input.get_shape().as_list()
        n_channels  = np.prod(input_shape[:-1])
        x_reshape   = tf.reshape(input,[n_channels,1,input_shape[-1]])

        outputs = list()
        for i in range(len(self.W)):
            # shape of W is (width,inchannel,outchannel)
            width,in_c,out_c = self.W[i].get_shape().as_list()
            amount_l = np.int32(np.floor((width-1)/2))
            amount_r = np.int32(width-1-amount_l)
            x_pad    = tf.pad(input,paddings=[[0,0],[0,0],
                            [amount_l,amount_r]],mode='SYMMETRIC')

            conv       = self.apply_filter_bank(x_pad,self.W[i])
            conv_shape = conv.shape.as_list()
            modulus    = tf.sqrt(tf.square(conv[:,:out_c])
                                        +tf.square(conv[:,out_c:]))
            new_shape  = input_shape[:-1]+[out_c,input_shape[-1]//self.strides]
            outputs.append(tf.reshape(modulus,new_shape))
        if len(self.W)>1:
            output = tf.concat(outputs,-2)
        else:
            output = outputs[0]
        return output


    def apply_filter_bank(self,input,W):
            filter_bank = tf.concat([tf.real(W),tf.imag(W)],2)
            return tf.nn.conv1d(input,filter_bank,stride=self.strides,
                                        padding='VALID',data_format='NCW')
    def init_mp(self, m, p, init):
        if m is not None and p is not None:
            self._m = m
            self._p = p
        else:
            if init=='gabor':
                window = np.hamming(self.K)
                m_real = np.cos(np.arange(self.K) * np.pi)*window
                m_imag = np.zeros(self.K)
                p_real = np.zeros(self.K)
                p_imag = np.cos(np.arange(self.K) * np.pi)*window
            elif init=='random':
                m_real = np.random.randn(self.K)/np.sqrt(self.K)
                m_imag = np.random.randn(self.K)/np.sqrt(self.K)
                p_real = np.random.randn(self.K)
                p_imag = np.random.randn(self.K)
            m = np.stack([m_real,m_imag]).astype('float32')
            p = np.stack([p_real,p_imag]).astype('float32')

            if self.hilbert:
                self._m = Variable(m[0], trainable=self.trainable_filters, name='m')
                self._p = Variable(p[0], trainable=self.trainable_filters, name='p')
            else:
                self._m = Variable(m, trainable=self.trainable_filters, name='m')
                self._p = Variable(p, trainable=self.trainable_filters, name='p')

        self.add_param(self._m)
        self.add_param(self._p)

        # Boundary Conditions and centering
        mask    = np.ones(self.K, dtype=np.float32)
        mask[0], mask[-1] = 0, 0
        m_null  = self._m - tf.reduce_mean(self._m[...,1:-1],axis=-1,
                                                    keepdims=True)
        self.m  = m_null*mask
        self.p  = self._p*mask


    def init_filters(self,start=0,end=-1):
        """
        Method initializing the filter-bank
        """

        # ------ TIME SAMPLING
        # add an extra octave if learnable scales (to go to lower frequency)
        # Define the integer time grid (time sampling) 
        length    = int(self.K*2**(self._scales[end-1]
                                        +int(self.trainable_scales)))
        time_grid = tf.linspace(np.float32(-(length//2)), 
                                    np.float32(length//2), length)

        # ------ FILTER-BANK
        if self.hilbert:
            filters_real = utils.hermite_interp(time_grid, 
                        self.all_knots[start:end], self.m, self.p, True)
            filters_fft = tf.spectral.rfft(filters)
            filters = tf.ifft(tf.concat([filters_fft,
                                            tf.zeros_like(filters_fft)],1))
        else:
            filters = utils.hermite_interp(time_grid, self.all_knots[start:end],
                                            self.m, self.p,False)
            filters = tf.complex(filters[0],filters[1])

        W = tf.expand_dims(tf.transpose(filters),1)
        W.set_shape((length,1,len(self._scales[start:end])))
        return W


