import tensorflow as tf
import numpy as np
from . import Layer



class RandomCrop(Layer):
    def __init__(self, incoming, crop_size,training):
        super().__init__(incoming)
        if self.data_format=='NCHW':
            images       = tf.transpose(incoming.output,[0,2,3,1])
            images_shape = [incoming.out_shape[0],incoming.out_shape[2],
                    incoming.out_shape[3],incoming.out_shape[1]]
        else:
            images       = incoming.output
            images_shape = incoming.out_shape
        indices_H      = tf.cast(tf.floor(tf.random_uniform((images_shape[0],),maxval=np.float32(images_shape[1]-crop_size[0]+1))),tf.int32)
        indices_W      = tf.cast(tf.floor(tf.random_uniform((images_shape[0],),maxval=np.float32(images_shape[2]-crop_size[1]+1))),tf.int32)
        random_patches = tf.extract_image_patches(images,[1,crop_size[0],crop_size[1],1],strides=[1,1,1,1],rates=[1,1,1,1],padding='VALID')
        selected_patches= tf.gather_nd(random_patches,tf.stack([tf.range(incoming.out_shape[0]),indices_H,indices_W],1))
        selected_images= tf.reshape(selected_patches,[images_shape[0],crop_size[0],crop_size[1],images_shape[-1]])
        center_images  = tf.gather_nd(random_patches,tf.stack([tf.range(incoming.out_shape[0]),
            tf.fill((images_shape[0],),np.int32((images_shape[1]-crop_size[0]+1)//2)),
                tf.fill((images_shape[0],),np.int32((images_shape[2]-crop_size[1]+1)//2))],1))
        output = tf.cond(training,lambda :selected_images,lambda :center_images)
        if self.data_format=='NCHW':
            self.output = tf.transpose(output,[0,3,1,2])
            self.out_shape = [images_shape[0],images_shape[3],crop_size[0],crop_size[1]]
        else:
            self.output = output
            self.out_shape = [images_shape[0],crop_size[0],crop_size[1],images_shape[3]]




class RandomFlipUpDown(Layer):
    def __init__(self, incoming, training):
        super().__init__(incoming)
        prob    = tf.random_uniform((incoming.out_shape[0],))
        up_down_= tf.less(prob,0.5)
        if self.data_format=='NCHW':
            images = tf.where(up_down_,tf.reverse(incoming.output,[2]),incoming.output)
        else:
            images = tf.where(up_down_,tf.reverse(incoming.output,[1]),incoming.output)
        self.output       = tf.cond(training,lambda :images,lambda :incoming.output)
        self.out_shape = incoming.out_shape



class RandomRot90(Layer):
    def __init__(self, incoming, training):
        super().__init__(incoming)
        prob   = tf.random_uniform((incoming.out_shape[0],))*3
        left_  = tf.less(prob,1)
        right_ = tf.greater(prob,2)
        if self.data_format=='NCHW':
            left_rot    = tf.transpose(incoming.output,[0,1,3,2])
            left_images = tf.where(left_,left_rot,incoming.output)
            images      = tf.where(right_,tf.reverse(left_rot,[-1]),left_images)
        else:
            left_rot    = tf.transpose(incoming.output,[0,1,3,2])
            left_images = tf.where(left_,left_rot,incoming.output)
            images      = tf.where(right_,tf.reverse(left_rot,[2]),left_images)
        self.output       = tf.cond(training,lambda :images,lambda :incoming.output)
        self.out_shape = incoming.out_shape




class RandomFlipLeftRight(Layer):
    def __init__(self, incoming, training):
        super().__init__(incoming)
        prob        = tf.random_uniform((incoming.out_shape[0],))
        left_right_ = tf.less(prob,0.5)
        if self.data_format=='NCHW':
            images = tf.where(left_right_,tf.reverse(incoming.output,[3]),incoming.output)
        else:
            images = tf.where(up_down_,tf.reverse(incoming.output,[2]),incoming.output)
        self.output       = tf.cond(training,lambda :images,lambda :incoming.output)
        self.out_shape = incoming.out_shape



