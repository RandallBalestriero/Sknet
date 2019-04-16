import tensorflow as tf
import tensorflow.contrib.layers as tfl
from .normalize import BatchNorm as bn
import numpy as np
from . import Op


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
    def __init__(self,incoming,filters,W = tfl.xavier_initializer(),
                    b = tf.zeros, strides=1, pad='valid',
                    mode='CONSTANT', name='', W_func = tf.identity,
                    b_func = tf.identity,*args,**kwargs):
        with tf.variable_scope("Conv2D") as scope:
            self.scope_name = scope.original_name_scope
            self.mode = mode
            if np.isscalar(strides):
                self.strides = [strides,strides]
            else:
                self.strides = strides
                
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
            if type(W)!=Variable:
                W = Variable(W, name='conv2dlayer_W_'+name)
            self._W = W(w_shape)
            self.W  = W_func(self._W)
            # Initialize b
            if b is None:
                self._b = None
                self.b  = None
                self._params = [self._W]
            else:
                if type(b)!=Variable:
                    b = Variable(b, name='conv2dlayer_b_'+name)
                self._b = b((1,filters[0],1,1))
                self.b  = b_func(self._b)
                self._params = [self._W,self._b]
    
            super().__init__(incoming)

    def forward(self,input, *args,**kwargs):
        if self.to_pad:
            padded = tf.pad(input,[[0,0],[0,0],[self.p[0]]*2,
                                [self.p[1]]*2],mode=self.mode)
        else:
            padded = input
        Wx = tf.nn.conv2d(padded,self.W,
                strides=[1,self.strides[0],self.strides[1],1],padding='VALID',
                data_format="NCHW")
        if self.b is None:
            return Wx
        else:
            return Wx+self.b




#######################################
#
#       1D spline
#






def complex_hermite_interp(t, all_knots, m, p):
    # Create it here for graph bugs
    M = tf.constant(np.array([[1, 0, -3, 2],
               [0, 0, 3, -2],
               [0, 1, -2, 1],
               [0, 0, -1, 1]]).astype('float32'))
    # Concatenate coefficients onto knots 0:-1 and 1:end
    xx = tf.stack([all_knots[:,:-1], all_knots[:,1:]], axis=2) # (SCALES KNOTS-1 2)
    mm = tf.stack([m[:,:-1], m[:,1:]], axis=2)  # (2 KNOTS-1 2)
    pp = tf.stack([p[:,:-1], p[:,1:]], axis=2)  # (2 KNOTS-1 2)

    y  = tf.concat([mm, pp], axis=2)            # (2 KNOTS-1 4)

    ym   = tf.einsum('iab,bc->iac',y, M)        # (KNOTS-1 4)

    x_n  = (t-tf.expand_dims(xx[:,:,0],2))/tf.expand_dims(xx[:,:,1]-xx[:,:,0],2) #(SCALES KNOTS-1,t)
    mask = tf.cast(tf.logical_and(tf.greater_equal(x_n, 0.), tf.less(x_n, 1.)), tf.float32)
    x_p  = tf.pow(tf.expand_dims(x_n,-1), [0,1,2,3])
    yi   = tf.einsum('irf,srtf->ist',ym,x_p*tf.expand_dims(mask,-1))
    return yi




def real_hermite_interp(t, all_knots, m, p):
    # Create it here for graph bugs
    M = tf.constant(np.array([[1, 0, -3, 2],
               [0, 0, 3, -2],
               [0, 1, -2, 1],
               [0, 0, -1, 1]]).astype('float32'))
    # Concatenate coefficients onto knots 0:-1 and 1:end
    xx = tf.stack([all_knots[:,:-1], 
                all_knots[:,1:]], axis=2)#(SCALES KNOTS-1 2)
    mm = tf.stack([m[:-1], m[1:]], axis=1)  # (KNOTS-1 2)
    pp = tf.stack([p[:-1], p[1:]], axis=1)  # (KNOTS-1 2)

    y  = tf.concat([mm, pp], axis=1)        # (KNOTS-1 4)

    ym   = tf.matmul(y, M)                  # (KNOTS-1 4)

    x_n  = (t-tf.expand_dims(xx[:,:,0],2))/tf.expand_dims(xx[:,:,1]-xx[:,:,0],2) #(SCALES KNOTS-1,t)
    mask = tf.cast(tf.logical_and(tf.greater_equal(x_n, 0.), 
                                        tf.less(x_n, 1.)), tf.float32)
    x_p  = tf.pow(tf.expand_dims(x_n,-1), [0,1,2,3])
    yi   = tf.einsum('rf,srtf->st',ym,x_p*tf.expand_dims(mask,-1))
    return yi



class WaveletTransform:
    """Learnable scattering network layer."""

    def __init__(self, x, J, Q, K, strides, init='gabor',
                 trainable_scales=True, trainable_knots=True, 
                 trainable_filter=True, hilbert=False, W=None,
                 padding='valid',W_type='dense',**kwargs):

        # Attribution
        self.trainable_scales = trainable_scales
        self.trainable_knots  = trainable_knots
        self.trainable_filter = trainable_filter
        self.hilbert          = hilbert
        # Input shape
        input_shape = x.get_shape().as_list()
        n_channels  = np.prod(input_shape[:-1])

        # Pad input
        x_reshape = tf.reshape(x,[n_channels,1,input_shape[-1]])
        if padding=='same':
            amount = np.int32(np.floor((self.filter_samples-1)/2))
            x_pad = tf.pad(x_reshape,paddings=[[0,0],[0,0],[amount]*2],
                                    mode='SYMMETRIC')
        elif padding=='valid':
            x_pad = x_reshape

        # Create filter bank (samples,1,J*Q) or (#,samples,1,J*Q/#)
        # 
        self.W = init_filters(J,Q,K)

        if hasattr(self.W,'__len__'):
            outputs = list()
            for w in self.W:
                if padding=='same':
                    amount = np.int32(np.floor((self.filter_samples-1)/2))
                    x_pad = tf.pad(x_reshape,paddings=[[0,0],[0,0],[amount]*2],
                                        mode='SYMMETRIC')
                elif padding=='valid':
                    x_pad = x_reshape

                conv = self.apply_filter_bank(x_pad,w,strides)
                jq   = w.shape.as_list()[-1]
                conv_shape = conv.shape.as_list()
                new_shape  = input_shape[:-1]+[jq,conv_shape[-1]]
                modulus    = tf.sqrt(tf.square(conv[:,:jq])
                                        +tf.square(conv[:,jq:]))
                outputs.append(tf.reshape(modulus,new_shape))
            output = tf.concat(outputs,-2)
        else:
            if padding=='same':
                amount = np.int32(np.floor((self.filter_samples-1)/2))
                x_pad = tf.pad(x_reshape,paddings=[[0,0],[0,0],[amount]*2],
                                    mode='SYMMETRIC')
            elif padding=='valid':
                x_pad = x_reshape

            conv = self.apply_filter_bank(x_pad,self.W,strides)
            conv_shape = conv.shape.as_list()
            new_shape = input_shape.[:-1]+[F,conv_shape[-1]]
            modulus   = tf.sqrt(tf.square(conv[:,:J*Q])+tf.square(conv[:,J*Q:]))
            output    = tf.reshape(modulus,new_shape)

        super().__init__(output)


        def apply_filter_bank(self,input,W,strides):
            filter_bank = tf.concat([tf.real(W),tf.imag(W)],2)
            conv      = tf.nn.conv1d(input,filter_bank,stride=strides,
                                        padding='VALID',data_format='NCW')
            return conv




def init_filters(self, j, q, k):
        # bin length of the filter bank, (k needs to be odd)

        # ------ TIME SAMPLING
        # add an extra octave if learnable scales 
        # if needs be to go lower frequency
        # Define the integer time grid (time sampling) = 
        # [-k*2**(j-1),...,-1,0,1,...,k*2**(j-1)]

        self.filter_samples = k*2**(j+int(self.trainable_scales))
        start_ = np.float32(-k*2**(j-1+int(self.trainable_scales)))
        end_   = np.float32(k*2**(j-1+int(self.trainable_scales)))
        time_grid = tf.lin_space(start_,end_, self.filter_samples)

        # ------ SCALES
        # we start with scale 1 for the nyquist and then increase 
        # the scale for the lower frequencies
        squeleton_scales = 2**(tf.range(j*q,dtype=tf.float32)/np.float32(q))
        scales_diff      = tf.Variable(tf.zeros(j*q), 
                            trainable=self.trainable_scales, name='scales')
        self.scales      = tf.contrib.framework.sort(squeleton_scales+scales_diff)

        # ------ KNOTS
        # We initialize with uniform spacing 
        grid = tf.lin_space(np.float32(-(k//2)),np.float32(k//2), k)
        self.knots_ = tf.Variable(grid, trainable=self.trainable_knots, 
                                                    name='knots')
        self.knots = tf.einsum('i,j->ij',self.scales,self.knots_)

        # ------ FILTERS
        # Interpolation init, add the boundary condition mask and remove the mean
        # filters of even indices are the real parts and odd indices are imaginary part
        if self.hilbert:
            # Create the parameters
            self.m = tf.Variable((np.cos(np.arange(k) * np.pi)*np.hamming(k)).astype('float32'), name='m', trainable=self.trainable_filter)
            self.p = tf.Variable((np.zeros(k)).astype('float32'), name='p', trainable=self.trainable_filter)
            # Boundary Conditions and centering
            mask              = np.ones(k, dtype=np.float32)
            mask[0], mask[-1] = 0, 0
            m_null            = self.m - tf.reduce_mean(self.m[1:-1])
            filters           = real_hermite_interp(time_grid, self.knots, m_null*mask, self.p*mask)
            # Renorm and set filter-bank
            filters_renorm    = filters/tf.reduce_max(filters,1,keepdims=True)
            filters_fft       = tf.spectral.rfft(filters_renorm)
            self.filter_bank  = tf.ifft(tf.concat([filters_fft,tf.zeros_like(filters_fft)],1))
        else:
            # Create the parameters
            self.m = tf.Variable(np.stack([np.cos(np.arange(k) * np.pi)*np.hamming(k),
                    np.zeros(k)*np.hamming(k)]).astype('float32') ,
                    name='m',trainable=self.trainable_filter)
            self.p = tf.Variable(np.stack([np.zeros(k),
                    np.cos(np.arange(k) * np.pi)*np.hamming(k)]).astype('float32'),
                    name='p',trainable=self.trainable_filter)
            # Boundary Conditions and centering
            mask              = np.ones((1,k), dtype=np.float32)
            mask[0,0], mask[0,-1] = 0, 0
            m_null            = self.m - tf.reduce_mean(self.m[:,1:-1],axis=1,keepdims=True)
            filters           = complex_hermite_interp(time_grid, self.knots, m_null*mask, self.p*mask)
            # Renorm and set filter-bank
            filters_renorm    = filters/tf.reduce_max(filters,2,keepdims=True)
            self.filter_bank  = tf.complex(filters_renorm[0],filters_renorm[1])


