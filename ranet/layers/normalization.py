import tensorflow as tf


def batch_normalization(tensor,axis,training,beta_initializer = tf.zeros, gamma_initializer=tf.ones ,center=True,scale=True,name='batch_normalization_layer',epsilon=1e-6,decay=0.99):
    input_shape     = tensor.get_shape().as_list()
    shape_          = [s if i not in axis else 1 for i,s in enumerate(input_shape)]
    beta            = tf.Variable(beta_initializer(shape_),trainable=center,name=name+'_beta')
    gamma           = tf.Variable(gamma_initializer(shape_),trainable=scale,name=name+'_gamma')
    moving_mean     = tf.Variable(tf.zeros(shape_),trainable=False,name=name+'_movingmean')
    moving_var      = tf.Variable(tf.ones(shape_),trainable=False,name=name+'_movingvar')
    cpt             = tf.Variable(tf.ones(1),trainable=False,name=name+'cpt')
    moments         = tf.nn.moments(tensor,axes=axis,keep_dims=True)
    coeff           = (cpt-1.)/cpt
    update_mean     = tf.assign(moving_mean,tf.cond(training,lambda :moving_mean*decay+moments[0]*(1-decay),lambda :moving_mean))
    update_var      = tf.assign(moving_var,tf.cond(training,lambda :moving_var*decay+moments[1]*(1-decay),lambda :moving_var))
    update_cpt      = tf.assign_add(cpt,tf.cond(training,lambda :tf.ones(1),lambda :tf.zeros(1)))
    update_ops      = tf.group(update_mean,update_var,update_cpt)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,update_ops)
    bias    = -(gamma*tf.cond(training,lambda :moments[0],lambda :moving_mean))/tf.sqrt(tf.cond(training,lambda :moments[1],lambda :moving_var)+epsilon)+beta
    scaling = gamma/tf.sqrt(tf.cond(training,lambda :moments[1],lambda :moving_var)+epsilon)
    return scaling,bias





