from tfwavelets.dwtcoeffs import haar, db3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from .utility import *
from . import custom_nodes as nodes

class Shifting(tf.keras.layers.Layer):
    def __init__(self):
        def layer(inp):
            return tfa.image.translate(inp[0],inp[1],"BILINEAR")
        self.shift_layer = tf.keras.layers.Lambda(layer)

    def restack(self, in1, in2):
        columns = []

        for i,col in enumerate(tf.unstack(in2, axis=1)):
            columns.extend([in1[:,i], col])
        columns.append(in1[:,i+1])
        return tf.stack(columns,axis=1)

    def __call__(self, input,shift):
        odd = True
        if input[0,0,0,0] ==0:
            odd = False
        substack1 = input[:,::2]
        substack2 = input[:,1::2]
        if odd:
            substack2 = self.shift_layer((substack2, shift))
        else:
            substack1 = self.shift_layer((substack1, shift))
        substack1 = tf.cast(substack1,tf.float64)
        substack2 = tf.cast(substack2,tf.float64)
        return self.restack(substack1, substack2)


class CompressedSensing(tf.keras.layers.Layer):
    def __init__(self,*args, **kwargs):
        super(CompressedSensing, self).__init__(*args, **kwargs, dtype="float32")

        #done: gaussian as initial value
        self.psf = get_psf(180, 100)  # todo: sigma px_size
        #
        self.mat = tf.Variable(initial_value=self.psf_initializer(), dtype=tf.float32, trainable=False)
        self.mu = tf.Variable(initial_value=np.ones((1)), dtype=tf.float32, trainable=False)
        self.lam = tf.Variable(initial_value=np.ones((1)), dtype=tf.float32, name="lambda", trainable=True)*0.005#was0.005
        self.t = tf.Variable(initial_value=np.ones((1)),dtype=tf.float32, trainable=False)
        #dense = lambda x: tf.sparse.to_dense(tf.SparseTensor(x[0], x[1], tf.shape(x[2], out_type=tf.int64)))
        #self.sparse_dense = tf.keras.layers.Lambda(dense)
        self.y = tf.Variable(initial_value=np.zeros((5329,3)), dtype=tf.float32, trainable=False)[tf.newaxis, :]
        #self.result = tf.Variable(np.zeros((73,73,3)), dtype=tf.float32, trainable=False)[tf.newaxis, :]
        self.tcompute = tf.keras.layers.Lambda(lambda t: (1+tf.sqrt(1+4*tf.square(t)))/2)
        self.flatten= tf.keras.layers.Flatten()
        self.matmul = tf.keras.layers.Lambda(lambda x: tf.keras.backend.dot(x,x))


    def update_psf(self, sigma, px_size):
        self.psf = get_psf(sigma, 100)
        self.mat = self.psf_initializer()

    def psf_initializer(self):
        mat = create_psf_matrix(9, 8, self.psf)
        return mat

    def softthresh2(self, input, lam):
        one = tf.keras.activations.relu(input-lam, threshold=0)
        two = tf.keras.activations.relu(-input-lam, threshold=0)
        return one-two

    def softthresh(self, input, lam):
        one = simulate_where_add(input, lam, tf.constant([np.inf],dtype=tf.float32), -lam, self.sparse_dense)
        two = simulate_where_add(input, tf.constant([-np.inf],dtype=tf.float32), -lam, lam, self.sparse_dense)
        return one+two

    #@tf.function
    def __call__(self, input):
        #done: fista here
        #input = tf.cast(input, tf.float32)
        inp = tf.unstack(input, axis=-1)
        y_n = tf.unstack(self.y, axis=-1)
        # print("a=" + str(len(inp)))
        # print("b=" +str(len(y_n)))
        r = []
        for i in range(len(inp)):
        #     x = inp[i]
            im = self.flatten(inp[i])
            y_new_last_it = tf.zeros_like(y_n[i])
            y_tmp = y_n[i]
            t = tf.constant((1.0),dtype=tf.float32)
            for j in range(100):
                re =tf.linalg.matvec(self.mat, im- tf.linalg.matvec(tf.transpose(self.mat), y_tmp))

                w = y_tmp+1/self.mu*re
                #if test:
                y_new = self.softthresh2(w, self.lam/self.mu)#todo: not exactly the same as softthresh
                # else:
                #     y_new = self.softthresh(w, self.lam/self.mu)
                #y_new = w#tf.cast(w, tf.float64)
                t_n = self.tcompute(t)
                y_tmp = y_new+ (self.t-1)/t_n*(y_new-y_new_last_it)
                y_new_last_it = y_new
                t = t_n
        #x = tf.tensordot(self.mat, input[:,:,:,0], 1)
            #r.append(tf.einsum('nm,im->in', self.mat, im))
            r.append(y_new)

        #b = tf.cast(y_n[0], tf.float64)
        return tf.stack(r,axis=-1)




