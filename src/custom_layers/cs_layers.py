import tensorflow as tf
from src.utility import get_psf,create_psf_matrix,old_psf_matrix
from src.custom_layers.utility_layer import downsample,upsample
from tensorflow.keras import initializers

class CompressedSensingInception(tf.keras.layers.Layer):
    def __init__(self,*args, iterations=100, **kwargs ):
        super(CompressedSensingInception, self).__init__(*args, **kwargs, dtype="float32")
        #define filters for path x
        #for all paths
        self.batch_norm0 = tf.keras.layers.BatchNormalization()

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        #done: estimate mu and sigma for cs layer?
        #for x path
        self.cs = CompressedSensing()
        self.cs.iterations = iterations
        self.reshape = tf.keras.layers.Reshape((72, 72, 3), )#done: compare low iteration with high iteration and implement additional loss
        #self.padding = tf.keras.layers.Lambda(lambda x:  tf.pad(x, paddings, "REFLECT"))
        self.convolution1x1_x1 = tf.keras.layers.Conv2D(3,(1,1), activation=tf.keras.layers.LeakyReLU(),
                                                        kernel_initializer=initializers.truncated_normal(mean=1.0, stddev=0.3),
                                                        kernel_constraint=tf.keras.constraints.non_neg())#reduce dim to 1 for cs max intensity proj? padding doesnt matter
        self.convolution1x1_x2 = tf.keras.layers.Conv2D(8,(1,1), activation=tf.keras.layers.LeakyReLU())#reduce dimension after max pooling
        self.convolution5x1_x = tf.keras.layers.Conv2D(16,(5,1),strides=(2,2), activation=None,padding="same")#reduce dim by 2 todo: prio2 include asymetric
        self.convolution1x5_x2 = tf.keras.layers.Conv2D(32,(1,5),strides=(2,2), activation=None,padding="same")#reduce dim by 2 todo: prio2 include asymetric
        self.convolution5x5_x3 = tf.keras.layers.Conv2D(64,(5,5),strides=(2,2), activation=None,padding="same")#reduce dim by 2 todo: prio2 include asymetric

        #self.max_pooling_x1 = tf.keras.layers.MaxPooling2D(pool_size=(4,4),strides=(4,4),padding="same")#reduce dimension todo try not to use max pooling...
        #self.max_pooling_x2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2),padding="same")#reduce dimension
        self.x_path = [self.convolution1x1_x1,#todo: normalize input here
                        #self.batch_norm1,
                       self.cs,
                       self.reshape,  #72x72x1
                       self.convolution5x1_x,  #72x72x1
                       self.convolution1x5_x2,  # 72x72x1
                       self.convolution5x5_x3,  # 72x72x1


                       #self.max_pooling_x1,#18x18x8
                       self.convolution1x1_x2,  #18x18x2
                       #self.max_pooling_x2
                       ]#9x9x2
        #define filters for path y
        self.convolution1x7_y1 = tf.keras.layers.Conv2D(32,(1,7), activation=None, padding="same")
        self.convolution7x1_y2 = tf.keras.layers.Conv2D(32,(7,1), activation=None, padding="same")
        self.convolution1x1_y1 = tf.keras.layers.Conv2D(32,(1,1), activation=tf.keras.layers.LeakyReLU())#todo: leaky relu?
        self.y_path = [
                       self.convolution1x7_y1,
                       self.convolution7x1_y2,
                        self.convolution1x1_y1,
]
        #define filters for path w
        self.convolution1x1_w1 = tf.keras.layers.Conv2D(1,(1,1), activation=tf.keras.layers.LeakyReLU())

        self.dropout_layer = tf.keras.layers.SpatialDropout2D(rate=0.2)

        #defince filters for path z micro u net
        #todo: prio1 include skips with concat
        self.down1 = downsample(12,5,strides=3,apply_batchnorm=True)
        self.down2 = downsample(24,5,strides=3)#todo: concat sigma here? concat sigma in extra z-path
        self.hidden1 = tf.keras.layers.Dense(24,kernel_initializer=initializers.RandomNormal(mean=-1,stddev=2),bias_initializer=initializers.Zeros())
        self.hidden2 = tf.keras.layers.Dense(12,kernel_initializer=initializers.RandomNormal(mean=-1,stddev=2),bias_initializer=initializers.Zeros())
        self.hidden3 = tf.keras.layers.Dense(1,kernel_initializer=initializers.RandomNormal(mean=-1,stddev=2),bias_initializer=initializers.Zeros(),
                                             kernel_constraint=tf.keras.constraints.min_max_norm(min_value=-1, max_value=1,))

        self.lam_activation = tf.keras.layers.Lambda(lambda x: 0.1*tf.keras.activations.sigmoid(x))

        # self.z_down1 = downsample(16,3,strides=3,apply_batchnorm=True)
        # self.z_down2 = downsample(32,3,strides=3)#todo: concat sigma here? concat sigma in extra z-path
        # self.z_up1 = upsample(64,3,strides=3)
        # self.z_up2 = upsample(128,3,strides=3)


        #defince output layer
        self.concat = tf.keras.layers.Concatenate(axis=-1)


    def __call__(self, input, training=False, test=False):
        outputs = []
        w = self.convolution1x1_w1(input)
        outputs.append(w)

        z = input
        z = self.batch_norm0(z)
        z = self.down1(z)
        z = self.down2(z)
        z = tf.keras.layers.Flatten()(z)
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.hidden3(z)
        param = self.lam_activation(z)
        #print(z.numpy())

        x = input#todo: normalize input
        x = x/(0.001+tf.reduce_max(tf.abs(x),axis=[1],keepdims=True))
        x = self.x_path[0](x)

        x = self.x_path[1](x, param*0.1)
        for i,layer in enumerate(self.x_path[2:]):
            x = layer(x)
            if i==0:
                cs_out = x#todo: additional loss with mat mul

        outputs.append(x)


        # #todo: input path y
        y1 = self.y_path[0](input)
        y2 = self.y_path[1](input)
        y = self.concat([y1,y2])
        #y = self.batch_norm2(y)
        y = self.y_path[2](y)
        outputs.append(y)


        # d1 = self.z_down1(input)
        # d2 = self.z_down2(d1)
        # z = self.concat([d2,tf.repeat(tf.cast(self.cs.sigma,tf.float32),100)[:,tf.newaxis,tf.newaxis,tf.newaxis]])
        # z = self.z_up1(z)
        # z = self.concat([d1,z])
        # z = self.z_up2(z)
        # outputs.append(z)


        #todo: input path w no path because it's pass through

        #todo: input path z

        #todo: add skip layers

        #todo: concat layer

        output = self.concat(outputs)#[w,x,y,z], )

        # if training:
        #    output = self.dropout_layer(output)
        if test:
            return output #todo: use cs out with inverse matrix to compute msq
        return output,cs_out



class CompressedSensing(tf.keras.layers.Layer):
    def __init__(self,*args, **kwargs):
        super(CompressedSensing, self).__init__(*args, **kwargs, dtype="float32")
        self._iterations = tf.constant(200, dtype=tf.int32)

        self._sigma =150
        self._px_size = 100
        self.matrix_update()#mat and psf defined outside innit

        #self.mu = tf.Variable(initial_value=tf.ones((1)), dtype=tf.float32, trainable=False)
        #self.lam = tf.Variable(initial_value=tf.ones((1))*0.005, dtype=tf.float32, name="lambda",
         #                      trainable=True)#was0.005
        #dense = lambda x: tf.sparse.to_dense(tf.SparseTensor(x[0], x[1], tf.shape(x[2], out_type=tf.int64)))
        #self.sparse_dense = tf.keras.layers.Lambda(dense)003
        self.y = tf.constant(tf.zeros((5184,3)), dtype=tf.float32)[tf.newaxis, :]#todo: use input dim
        #self.result = tf.Variable(np.zeros((73,73,3)), dtype=tf.float32, trainable=False)[tf.newaxis, :]
        self.flatten= tf.keras.layers.Flatten()
        self.tcompute = tf.keras.layers.Lambda(lambda t: (1+tf.sqrt(1+4*tf.square(t)))/2)
        self.matmul = tf.keras.layers.Lambda(lambda x: tf.keras.backend.dot(x,x))

        #todo: add reshape

    @property
    def px_size(self):
        return self._px_size

    @property
    def sigma(self):
        return self._sigma

    @px_size.setter
    def px_size(self, value):
        self._px_size = value
        self.matrix_update()


    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.matrix_update()

    @property
    def iterations(self):
        return self._iterations.numpy()

    @iterations.setter
    def iterations(self, value):
        self._iterations = tf.constant(value, dtype=tf.int32)

    def matrix_update(self):
        self.psf = get_psf(self.sigma, self._px_size)
        self.mat = tf.constant(self.psf_initializer(), dtype=tf.float32)#works in training


    def psf_initializer(self):
        print("setting sigma")

        #mat1 = create_psf_matrix(9, 8, self.psf)
        v = self._sigma/self._px_size
        mat = old_psf_matrix(9,9,72,72, v*9/(64),v*9/(64)).T#todo: WTF why is this like this???
        # mat1 /= mat1[0,0]
        # mat1 *= mat[0,0]
        return mat

    @tf.function
    def softthresh2(self, input, lam):
        one = tf.keras.activations.relu(input-lam, threshold=0)
        two = tf.keras.activations.relu(-input-lam, threshold=0)
        return one-two

    #@tf.function
    def __call__(self, input, param):
        lam = param#todo: changed to softplus
        mu = 1.0
        print(lam[0])
        inp = tf.unstack(input, axis=-1)
        y_n = tf.unstack(self.y, axis=-1)
        r = []
        for i in range(len(inp)):
            im = self.flatten(inp[i])
            y_new_last_it = tf.zeros_like(y_n[i])#will be broadcasted
            y_tmp = y_n[i]
            t = tf.constant((1.0),dtype=tf.float32)
            for j in range(self.iterations):#todo iteration as variable
                re =tf.linalg.matvec(self.mat, im - tf.linalg.matvec(tf.transpose(self.mat), y_tmp))
                w = y_tmp+1/mu*re
                y_new = self.softthresh2(w, lam/mu)#todo: not exactly the same as softthresh
                t_n = self.tcompute(t)
                y_tmp = y_new+ (t-1)/t_n*(y_new-y_new_last_it)
                y_new_last_it = y_new
                t = t_n
            r.append(y_new)

        return tf.stack(r,axis=-1)

class CompressedSensingConvolutional(tf.keras.layers.Layer):
    def __init__(self,*args, **kwargs):
        super(CompressedSensingConvolutional, self).__init__(*args, **kwargs, dtype="float32")
        self._iterations = tf.constant(200, dtype=tf.int32)
        self._sigma = 150
        self._px_size = 100

        self.convtrap1 = tf.keras.layers.Conv2DTranspose(3, (5,5), strides=(8,8),padding="same")
        self.conv2D = tf.keras.layers.Conv2D(3,(40,40), strides=(8,8),padding="same")
        self.tcompute = tf.keras.layers.Lambda(lambda t: (1+tf.sqrt(1+4*tf.square(t)))/2)

    def __call__(self, input, lam):
        mu = 1.0
        batch = tf.shape(input)[0]
        y_tmp = tf.zeros((batch,72,72,3))#todo: variable shape
        y_new_last_it = tf.zeros((batch,72,72,3))
        y_new= tf.zeros((batch,72,72,3))
        t = tf.constant((1.0), dtype=tf.float32)

        for i in range(self._iterations):
            re = self.convtrap1(input-self.conv2D(y_tmp))
            w = y_tmp-1/mu*re
            y_new = self.softthresh2(w, lam / mu)  # todo: not exactly the same as softthresh
            t_n = self.tcompute(t)
            y_tmp = y_new + (t - 1) / t_n * (y_new - y_new_last_it)
            y_new_last_it = y_new
            t = t_n
        return y_new

    @property
    def px_size(self):
        return self._px_size

    @property
    def sigma(self):
        return self._sigma

    @px_size.setter
    def px_size(self, value):
        self._px_size = value
        self.matrix_update()

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.matrix_update()


    @property
    def iterations(self):
        return self._iterations.numpy()

    @iterations.setter
    def iterations(self, value):
        self._iterations = tf.constant(value, dtype=tf.int32)

    def matrix_update(self):
        self.psf = get_psf(self.sigma, self._px_size)
        self.mat = tf.constant(self.psf_initializer(), dtype=tf.float32)#works in training


    def psf_initializer(self):
        print("setting sigma")

        #mat1 = create_psf_matrix(9, 8, self.psf)
        v = self._sigma/self._px_size
        mat = old_psf_matrix(9,9,72,72, v*9/(64),v*9/(64)).T#todo: WTF why is this like this???
        # mat1 /= mat1[0,0]
        # mat1 *= mat[0,0]
        return mat


    @tf.function
    def softthresh2(self, input, lam):
        one = tf.keras.activations.relu(input-lam[:,tf.newaxis,tf.newaxis], threshold=0)
        two = tf.keras.activations.relu(-input-lam[:,tf.newaxis,tf.newaxis], threshold=0)
        return one-two

