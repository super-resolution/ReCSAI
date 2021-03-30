import tensorflow as tf
from src.utility import get_psf,create_psf_matrix,old_psf_matrix
from src.custom_layers.utility_layer import downsample,upsample

class CompressedSensingInception(tf.keras.layers.Layer):
    def __init__(self,*args, iterations=100, **kwargs ):
        super(CompressedSensingInception, self).__init__(*args, **kwargs, dtype="float32")
        #define filters for path x
        #for all paths
        self.batch_norm = tf.keras.layers.BatchNormalization()
        #for x path
        self.cs = CompressedSensing()#todo: prio1 needs input dimension and update for sigma
        self.cs.iterations = iterations
        self.reshape = tf.keras.layers.Reshape((72, 72, 1), )
        paddings = tf.constant([[0,0],[1,0],[1,0],[0,0]])#expand size for even output
        #self.padding = tf.keras.layers.Lambda(lambda x:  tf.pad(x, paddings, "REFLECT"))
        self.convolution1x1_x1 = tf.keras.layers.Conv2D(1,(1,1), activation=tf.keras.layers.LeakyReLU())#reduce dim to 1 for cs max intensity proj? padding doesnt matter
        self.convolution1x1_x2 = tf.keras.layers.Conv2D(1,(1,1), activation=tf.keras.layers.LeakyReLU())#reduce dimension after max pooling
        self.convolution5x5_x = tf.keras.layers.Conv2D(8,(5,5), activation=None,padding="same")#reduce dim by 2 todo: prio2 include asymetric
        self.max_pooling_x1 = tf.keras.layers.MaxPooling2D(pool_size=(4,4),strides=(4,4),padding="same")#reduce dimension
        self.max_pooling_x2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2),padding="same")#reduce dimension
        self.x_path = [self.convolution1x1_x1,
                        self.cs,
                       self.reshape,#72x72x1
                       self.batch_norm,
                       self.convolution5x5_x,#72x72x1
                       self.max_pooling_x1,#18x18x8
                       self.convolution1x1_x2,#18x18x2
                       self.max_pooling_x2]#9x9x2
        #define filters for path y
        self.convolution1x7_y1 = tf.keras.layers.Conv2D(2,(1,7), activation=None, padding="same")
        self.convolution7x1_y2 = tf.keras.layers.Conv2D(4,(7,1), activation=None, padding="same")
        self.convolution1x1_y1 = tf.keras.layers.Conv2D(1,(1,1), activation=tf.keras.layers.LeakyReLU())#todo: leaky relu?
        self.y_path = [self.convolution1x7_y1,
                       self.convolution7x1_y2,
                       self.convolution1x1_y1,
                       self.batch_norm]
        #define filters for path w
        self.convolution1x1_w1 = tf.keras.layers.Conv2D(1,(1,1), activation=tf.keras.layers.LeakyReLU())

        #defince filters for path z micro u net
        #todo: prio1 include skips with concat
        self.down1 = downsample(12,3,strides=3,apply_batchnorm=True)
        self.down2 = downsample(24,3,strides=3)
        self.up1 = upsample(12,3,strides=3)
        self.up2 = upsample(1,3,strides=3)

        self.z_path_down= [self.down1,
                           self.down2]


        #defince output layer
        self.concat = tf.keras.layers.Concatenate(axis=-1)


    def __call__(self, input):
        #todo: split input in three..
        x = input
        for layer in self.x_path:
            x = layer(x)

        #todo: input path y
        y = input
        for layer in self.y_path:
            y = layer(y)

        #todo: input path w no path because it's pass through
        w = self.convolution1x1_w1(input)

        #todo: input path z
        z = input
        skips = []
        for layer in self.z_path_down:
            z = layer(z)
            skips.append(z)
        skip = skips[0]
        z = self.up1(z)
        z = self.concat([z,skip])
        z = self.up2(z)
        #todo: add skip layers

        #todo: concat layer
        output = self.concat([w,x,y,z], )
        return output



class CompressedSensing(tf.keras.layers.Layer):
    def __init__(self,*args, **kwargs):
        super(CompressedSensing, self).__init__(*args, **kwargs, dtype="float32")
        self._iterations = tf.constant(200, dtype=tf.int32)

        self._sigma =150
        self._px_size = 100
        self.matrix_update()#mat and psf defined outside innit

        self.mu = tf.Variable(initial_value=tf.ones((1)), dtype=tf.float32, trainable=False)
        self.lam = tf.Variable(initial_value=tf.ones((1)), dtype=tf.float32, name="lambda",
                               trainable=True)*0.003#was0.005
        #dense = lambda x: tf.sparse.to_dense(tf.SparseTensor(x[0], x[1], tf.shape(x[2], out_type=tf.int64)))
        #self.sparse_dense = tf.keras.layers.Lambda(dense)
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


    def update_psf(self, sigma, px_size):
        self.psf = get_psf(sigma, 100)
        self.mat = self.psf_initializer()

    def psf_initializer(self):
        #mat1 = create_psf_matrix(9, 8, self.psf)
        v = self._sigma/self._px_size
        mat = old_psf_matrix(9,9,72,72, v*9/(64),v*9/(64)).T#todo: WTF why is this like this???
        # mat1 /= mat1[0,0]
        # mat1 *= mat[0,0]
        return mat

    def softthresh2(self, input, lam):
        one = tf.keras.activations.relu(input-lam, threshold=0)
        two = tf.keras.activations.relu(-input-lam, threshold=0)
        return one-two

    #@tf.function
    def __call__(self, input):
        inp = tf.unstack(input, axis=-1)
        y_n = tf.unstack(self.y, axis=-1)
        r = []
        for i in range(len(inp)):
            im = self.flatten(inp[i])
            y_new_last_it = tf.zeros_like(y_n[i])
            y_tmp = y_n[i]
            t = tf.constant((1.0),dtype=tf.float32)
            for j in range(self.iterations):#todo iteration as variable
                re =tf.linalg.matvec(self.mat, im - tf.linalg.matvec(tf.transpose(self.mat), y_tmp))
                w = y_tmp+1/self.mu*re
                y_new = self.softthresh2(w, self.lam/self.mu)#todo: not exactly the same as softthresh
                t_n = self.tcompute(t)
                y_tmp = y_new+ (t-1)/t_n*(y_new-y_new_last_it)
                y_new_last_it = y_new
                t = t_n
            r.append(y_new)

        return tf.stack(r,axis=-1)