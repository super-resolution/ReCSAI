from src.custom_layers.cs_layers import CompressedSensing, CompressedSensingInception
#from tensorflow.keras.layers import *
import tensorflow as tf
from src.custom_layers.utility_layer import downsample,upsample


#todo: imitate yolo architecture and use 9x9 output grid
class CompressedSensingInceptionNet(tf.keras.Model):
    TYPE = 0
    def __init__(self):
        super(CompressedSensingInceptionNet, self).__init__()
        self.inception1 = CompressedSensingInception(iterations=10)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        #todo: extend depth?

        #self.inception2 = CompressedSensingInception(iterations=100)
        #self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.horizontal_path = [
            tf.keras.layers.Conv2D(256, (1, 1), activation=None, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Conv2D(128, (7, 1), activation=None, padding="same"),
            tf.keras.layers.Conv2D(64, (1, 7), activation=None, padding="same"),
            #tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),#todo dont do max pooling...
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32,(3,3), activation=None, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Conv2D(6, (3, 3), activation=None, padding="same"),
        ]#x,y,sigmax,sigmay,classifier

        def activation(inputs):
            inputs_list = tf.unstack(inputs,axis=-1)
            inputs_list[2] = tf.keras.activations.softmax(inputs_list[2], axis=[-1,-2])#last is classifier
            return tf.stack(inputs_list, axis=-1)

        def softplus_activation(inputs):#softplus activation
            inputs_list = tf.unstack(inputs, axis=-1)

            inputs_list[3] = tf.keras.activations.softplus(inputs_list[3])
            inputs_list[4] = tf.keras.activations.softplus(inputs_list[4])
            return tf.stack(inputs_list, axis=-1)
        self.error_activation = tf.keras.layers.Lambda(softplus_activation)


        def sigmoid_acitvaiton(inputs):
            inputs_list = tf.unstack(inputs, axis=-1)
            inputs_list[2] = tf.keras.activations.sigmoid(inputs_list[2])  # last is classifier
            return tf.stack(inputs_list, axis=-1)
        self.activation = tf.keras.layers.Lambda(sigmoid_acitvaiton)

    def __call__(self, inputs, training=False):
        x = inputs
        x, cs = self.inception1(x, training=training)
        x = self.batch_norm(x)
        # x = self.inception2(x, training=training)
        # x = self.batch_norm2(x)

        for layer in self.horizontal_path:
            x = layer(x)
        #todo: transpose

        x = self.activation(x)
        x = self.error_activation(x)
        if training:
            return x,cs
        else:
            return x

    @property
    def sigma(self):
        #if self.inception1.cs.sigma != self.inception2.cs.sigma:
        #    raise ValueError("sigma has to be identical in both layers")
        return self.inception1.cs.sigma

    @sigma.setter
    def sigma(self, value):
        self.inception1.cs.sigma = value
        #self.inception2.cs.sigma = value

    def compute_loss(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1
        ce = tf.keras.losses.BinaryCrossentropy()
        mask = truth[:,:,:,3:4]
        mask2 = truth[:,:,:,3]
        predict_masked = predict[:,:,:,0:2]*mask
        x = tf.constant([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])#[tf.newaxis,tf.newaxis,tf.newaxis,:]
        X,Y = tf.meshgrid(x,x)
        X_truth = (truth[:,:,:,0]+X)*mask2
        Y_truth = (truth[:,:,:,1]+Y)*mask2
        X_pred = (predict[:,:,:,0]+X)*mask2
        Y_pred = (predict[:,:,:,1]+Y)*mask2
        sigma = tf.abs(predict_masked - truth[:,:,:,0:2])
        #L2 = 0

        L2 = -tf.math.log(
            tf.reduce_sum(predict[:,:,:,2]/tf.reduce_sum(predict[:,:,:,2])
                          * tf.exp(-(tf.square(X_pred-X_truth)+tf.square(Y_pred-Y_truth)))))

        #L2 /= 81**2
        L2_sigma = l2(sigma, predict[:,:,:,3:5]*mask)
        BCE = ce(truth[:,:,:,2], predict[:,:,:,2],)
        STD = l2(self.sigma/100, predict[:,:,:,5])
        return 100*BCE + L2 + 3*L2_sigma + 3*STD


    def compute_decode_loss_dfp(self, truth, predict):
        pass

    def compute_loss_log_cs_out(self, truth, predict, cs_result, noiseless_truth):
        #todo: penaltize entries unequal to zero
        #todo: noiseless truth with mat mul
        l2 = tf.keras.losses.MeanSquaredError()
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1
        loss = 0
        inp_cs = tf.unstack(cs_result, axis=-1)
        inp_ns = tf.unstack(noiseless_truth, axis=-1)
        for i,j in zip(inp_cs, inp_ns):
            res = tf.linalg.matvec(tf.transpose(self.inception1.cs.mat), tf.keras.layers.Reshape((5184,), )(i), )
            loss += l1(tf.keras.layers.Reshape((9,9), )(res),j)


        ce = tf.keras.losses.BinaryCrossentropy()
        mask = truth[:,:,:,3:4]
        mask_reduced = truth[:,:,:,3]

        predict_masked = predict[:,:,:,0:2]*mask
        x = tf.constant([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])#[tf.newaxis,tf.newaxis,tf.newaxis,:]
        X,Y = tf.meshgrid(x,x)
        X_truth = (truth[:,:,:,0]+X)*mask_reduced
        Y_truth = (truth[:,:,:,1]+Y)*mask_reduced
        X_pred = (predict[:,:,:,0]+X)*mask_reduced
        Y_pred = (predict[:,:,:,1]+Y)*mask_reduced
        sigma = tf.abs(predict_masked - truth[:,:,:,0:2])
        L2 = 0
        # for i in range(9):
        #     for j in range(9):
        normed_truth = predict[:,:,:,2]*mask_reduced/(tf.reduce_sum(predict[:,:,:,2]*mask_reduced,axis=[1,2], keepdims=True))
        #todo: devide by covariance kernel
        cov = tf.exp(-tf.square(sigma-predict[:,:,:,3:5]))
        inner_loss = tf.reduce_sum(
            (normed_truth)/(2*3.14*predict[:,:,:,3]*predict[:,:,:,4])*
                          tf.exp(-1/2*(tf.square((X_pred-X_truth)/(predict[:,:,:,3]))+
                                       tf.square((Y_pred-Y_truth/(predict[:,:,:,4])))))
                ,axis=[1,2])
        loss_tensor = tf.math.log(tf.where(inner_loss>0,inner_loss,tf.ones_like(inner_loss)
            ))
        L2 = -tf.reduce_sum(loss_tensor*
            tf.reduce_sum(truth[:,:,:,2],axis=[1,2]))#todo: truth- tf.abs(normed_truth)*?



        L2_sigma = tf.reduce_sum(tf.math.log(1+tf.square(sigma- predict[:,:,:,3:5])))
        count_sigma = 0.001+tf.reduce_mean(predict[:,:,:,2]*(1-predict[:,:,:,2]))
        BCE = 10*ce(truth[:,:,:,2], predict[:,:,:,2],)#+ 15*tf.reduce_mean(predict[:,:,:,2]*(1-predict[:,:,:,2]))
        count_loss = +20*(tf.reduce_sum(tf.square(tf.reduce_sum(truth[:,:,:,2],[1,2])-tf.reduce_sum(predict[:,:,:,2]*mask_reduced, [1,2])))/count_sigma
                          -tf.math.log(tf.sqrt(count_sigma)))
        STD = l2(self.sigma/100, predict[:,:,:,5])
        return 2*L2 + 3*L2_sigma + 3*STD + loss + count_loss

    def compute_loss_log(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1
        ce = tf.keras.losses.BinaryCrossentropy()
        mask = truth[:,:,:,3:4]
        predict_masked = predict[:,:,:,0:2]*mask
        x = tf.constant([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])#[tf.newaxis,tf.newaxis,tf.newaxis,:]
        X,Y = tf.meshgrid(x,x)
        X_truth = truth[:,:,:,0]+X
        Y_truth = truth[:,:,:,1]+Y
        X_pred = predict[:,:,:,0]+X
        Y_pred = predict[:,:,:,1]+Y
        sigma = tf.abs(predict_masked - truth[:,:,:,0:2])
        L2 = 0
        # for i in range(9):
        #     for j in range(9):
        #normed_truth = predict[:,:,:,2]/tf.reduce_sum(predict[:,:,:,2],[1,2], keepdims=True)
        #todo: devide by covariance kernel
        cov = tf.exp(-tf.square(sigma-predict[:,:,:,3:5]))
        L2 = tf.reduce_mean(tf.math.log(
                          1+tf.square(X_pred-X_truth)/cov[:,:,:,0]+tf.square(Y_pred-Y_truth)/cov[:,:,:,1]))#todo: truth- tf.abs(normed_truth)*?

        L2_sigma = tf.reduce_sum(tf.math.log(1+tf.square(sigma- predict[:,:,:,3:5])))
        BCE = 10*ce(truth[:,:,:,2], predict[:,:,:,2],)+ 15*tf.reduce_mean(predict[:,:,:,2]*(1-predict[:,:,:,2]))
        count_loss = +20*tf.reduce_sum(tf.reduce_sum(truth[:,:,:,2],[1,2])-tf.reduce_sum(truth[:,:,:,2], [1,2]))
        STD = l2(self.sigma/100, predict[:,:,:,5])
        return 40*BCE + 2*L2 + 3*L2_sigma + 3*STD + count_loss

    def compute_loss_log_n_test(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1
        ce = tf.keras.losses.BinaryCrossentropy()
        mask = truth[:,:,:,3:4]
        predict_masked = predict[:,:,:,0:2]*mask
        x = tf.constant([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])#[tf.newaxis,tf.newaxis,tf.newaxis,:]
        X,Y = tf.meshgrid(x,x)
        X_truth = truth[:,:,:,0]+X
        Y_truth = truth[:,:,:,1]+Y
        X_pred = predict[:,:,:,0]+X
        Y_pred = predict[:,:,:,1]+Y
        sigma = tf.abs(predict_masked - truth[:,:,:,0:2])
        L2 = 0
        # for i in range(9):
        #     for j in range(9):
        normed_truth = predict[:,:,:,2]/tf.reduce_sum(predict[:,:,:,2],axis=[1,2], keepdims=True)
        #todo: devide by covariance kernel
        cov = tf.exp(-tf.square(sigma-predict[:,:,:,3:5]))
        L2 = -tf.math.log(tf.reduce_sum((normed_truth/(4*3.14*predict[:,:,:,3]*predict[:,:,:,4]))
                                       *tf.math.exp(-(tf.square(X_pred-X_truth)/tf.square(predict[:,:,:,3])
                                                      +tf.square(Y_pred-Y_truth)/tf.square(predict[:,:,:,4]))),axis=[1,2]))#todo: truth- tf.abs(normed_truth)*?
        L2_log = tf.reduce_sum(tf.math.log(
                          1+tf.square(X_pred-X_truth)/cov[:,:,:,0]+tf.square(Y_pred-Y_truth)/cov[:,:,:,1]))
        L2_sigma = tf.reduce_sum(tf.math.log(1+tf.square(sigma- predict[:,:,:,3:5])))
        BCE = ce(truth[:,:,:,2], predict[:,:,:,2],)#+ 15*tf.reduce_mean(predict[:,:,:,2]*(1-predict[:,:,:,2]))
        count_loss = tf.reduce_sum(tf.square(tf.reduce_sum(truth[:,:,:,2],[1,2])-tf.reduce_sum(predict[:,:,:,2], [1,2])))
        STD = l2(self.sigma/100, predict[:,:,:,5])
        return 1*BCE+tf.reduce_sum(L2)#+count_loss#tf.reduce_sum(L2)+BCE#count_loss

    def compute_loss_decode(self, truth,predict,_ ):
        l2 = tf.keras.losses.MeanSquaredError()
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1
        ce = tf.keras.losses.BinaryCrossentropy()
        #mask = truth[:,:,:,3:4]#truth is now in coordinates
        #mask2 = truth[:,:,:,3]


        x = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])  # [tf.newaxis,tf.newaxis,tf.newaxis,:]
        X, Y = tf.meshgrid(x, x)
        L2 = 0
        count = tf.zeros(tf.shape(truth)[0])
        for j in range(3):
            count += tf.where(truth[:,j,0] > 0,  tf.constant(1, dtype=tf.float32),
                        tf.constant(0.01, dtype=tf.float32))
        for i in range(3):
            L2 += tf.reduce_sum(tf.where(truth[:,i:i+1,0:1] > 0,-tf.math.log(tf.reduce_sum(
                tf.keras.activations.softmax(predict[:, :, :, 2], axis=[-1,-2]) /
                                 tf.math.sqrt(tf.math.sqrt((predict[:,:, :, 3])) *
                                               2 * 3.14 *
                                               tf.math.sqrt(predict[:,:, :, 4])
                                              )
            *tf.math.exp(-1/2*(
                                tf.square(
                                        predict[:,:, :, 0] + Y - truth[:,i:i+1,0:1])  # todo: test that this gives expected values
                                         / (0.001+predict[:,:, :, 3])
                                         + tf.square(predict[:,:, :, 1] + X - truth[:,i:i+1,1:2])
                                         / (0.001+predict[:, :, :, 4])
                                             )),axis=[1,2])) # todo: activation >= 0
                                        ,tf.constant(0, dtype=tf.float32)
                                         ,),
                            )

        sigma_c = 0.001+tf.reduce_sum(predict[:,:, :, 2] * (1 - predict[:, :,:, 2]),axis=[1,2])
        #i=0
        c_loss = tf.reduce_sum(3*tf.square(tf.reduce_sum(predict[:, :, :, 2], axis=[1,2])-count)/sigma_c-tf.math.log(tf.sqrt(2*3.14*sigma_c)))
        return  tf.reduce_mean(L2)+c_loss



class CompressedSensingCVNet(tf.keras.Model):
    TYPE = 0

    def __init__(self):
        super(CompressedSensingCVNet, self).__init__()
        self.cs_layer = CompressedSensing()
        self.reshape = tf.keras.layers.Reshape((72, 72, 3), input_shape=(72*72,3) )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        #todo: keep part of the down path
        self.down_path = [downsample(64,3,apply_batchnorm=True),
        downsample(128,3),
        downsample(256,3,apply_batchnorm=True),
                          ]
        #todo: concatnate here

        #todo: horizontal path

        self.horizontal_path = [
            tf.keras.layers.Conv2D(256, (1, 1), activation=None, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Conv2D(128, (7, 1), activation=None, padding="same"),
            tf.keras.layers.Conv2D(64, (1, 7), activation=None, padding="same"),
            #tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32,(3,3), activation=None, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Conv2D(6, (3, 3), activation=None, padding="same"),

        ]


        #todo: activation might not be that bad
        def softmax_activation(inputs):
            inputs_list = tf.unstack(inputs,axis=-1)
            inputs_list[2] = tf.keras.activations.softmax(inputs_list[2], axis=[-1,-2])#last is classifier
            return tf.stack(inputs_list, axis=-1)
        def sigmoid_activation(inputs):
            inputs_list = tf.unstack(inputs,axis=-1)
            inputs_list[2] = tf.keras.activations.sigmoid(inputs_list[2])#last is classifier
            return tf.stack(inputs_list, axis=-1)

        self.activation = tf.keras.layers.Lambda(sigmoid_activation)
        self.concat = tf.keras.layers.Concatenate()

    def __call__(self, inputs, training=False):
        x = self.cs_layer(inputs)
        x = self.reshape(x)#72x72
        x = self.batch_norm(x)
        for layer in self.down_path:
            x = layer(x)



        x = self.concat([x, inputs])
        for layer in self.horizontal_path:
            x = layer(x)

        x = self.activation(x)
        return x

    @property
    def sigma(self):
        return self.cs_layer.sigma

    @sigma.setter
    def sigma(self, value):
        self.cs_layer.sigma = value

    def compute_loss(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1
        ce = tf.keras.losses.BinaryCrossentropy()
        mask = truth[:,:,:,3:4]
        predict_masked = predict[:,:,:,0:2]*mask
        x = tf.constant([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])#[tf.newaxis,tf.newaxis,tf.newaxis,:]
        X,Y = tf.meshgrid(x,x)
        X_truth = truth[:,:,:,0]+X
        Y_truth = truth[:,:,:,1]+Y
        X_pred = predict[:,:,:,0]+X
        Y_pred = predict[:,:,:,1]+Y
        sigma = tf.abs(predict_masked - truth[:,:,:,0:2])
        L2 = 0
        # for i in range(9):
        #     for j in range(9):
        L2 += tf.math.log(
            tf.reduce_sum(predict[:,:,:,2]/tf.reduce_sum(predict[:,:,:,2])
                          * tf.exp(tf.square(X_pred-X_truth)+tf.square(Y_pred-Y_truth))))

        L2 /= 81
        L2_sigma = l2(sigma, predict[:,:,:,3:5])
        BCE = ce(truth[:,:,:,2], predict[:,:,:,2],)
        STD = l2(self.sigma/100, predict[:,:,:,5])
        return 100*BCE + L2 + 3*L2_sigma + 3*STD


    def compute_loss_log(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        l1 = tf.keras.losses.MeanAbsoluteError()#todo: switched to l1
        ce = tf.keras.losses.BinaryCrossentropy()
        mask = truth[:,:,:,3:4]
        predict_masked = predict[:,:,:,0:2]*mask
        x = tf.constant([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])#[tf.newaxis,tf.newaxis,tf.newaxis,:]
        X,Y = tf.meshgrid(x,x)
        X_truth = truth[:,:,:,0]+X
        Y_truth = truth[:,:,:,1]+Y
        X_pred = predict[:,:,:,0]+X
        Y_pred = predict[:,:,:,1]+Y
        sigma = tf.abs(predict_masked - truth[:,:,:,0:2])
        L2 = 0
        # for i in range(9):
        #     for j in range(9):
        L2 = tf.reduce_sum(tf.math.log(
                          tf.square(X_pred-X_truth)+tf.square(Y_pred-Y_truth)))

        L2_sigma = tf.reduce_sum(tf.math.log(sigma, predict[:,:,:,3:5]))
        BCE = ce(truth[:,:,:,2], predict[:,:,:,2],)
        STD = l2(self.sigma/100, predict[:,:,:,5])
        return 100*BCE + L2 + 3*L2_sigma + 3*STD

    def compute_loss_old(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        ce = tf.keras.losses.BinaryCrossentropy()
        mask = truth[:,:,:,3:4]
        L2 = l2(predict[:,:,:,0:2]*mask, truth[:,:,:,0:2])
        BCE = ce(truth[:,:,:,2], predict[:,:,:,2],)
        return BCE + 8*L2
        #todo: build truth image and compute loss
        #todo: select loc compute coordinates
        #todo: compute loss...



class CompressedSensingNet(tf.keras.Model):
    TYPE = 1
    def __init__(self):
        super(CompressedSensingNet, self).__init__()
        self.cs_layer = CompressedSensing()
        self.reshape = tf.keras.layers.Reshape((72, 72, 3), )

        # self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same")
        # self.pooling_3 = tf.keras.layers.MaxPooling2D((3, 3))
        # self.pooling_2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        # self.batch_norm2 = tf.keras.layers.BatchNormalization()
        # self.batch_norm3 = tf.keras.layers.BatchNormalization()
        #
        #
        # self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")
        # self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same")
        # self.conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="same")

        self.conv1 = downsample(32,3,apply_batchnorm=True)
        self.conv2 = downsample(64,3)
        self.conv3 = downsample(128,3,apply_batchnorm=True)
        self.conv4 = downsample(256,3,apply_batchnorm=True)


        self.dense1 = tf.keras.layers.Dense(64, activation="relu")#todo: add l1 regularization
        self.dense2 = tf.keras.layers.Dense(32, activation="relu")#todo: add l1 regularization
        self.dense3 = tf.keras.layers.Dense(16, activation="relu")#todo: add l1 regularization
        self.dense4 = tf.keras.layers.Dense(9)

        def activation(tensor):
            x = tf.unstack(tensor, axis=-1)
            for i in range(3):
                x[6+i] = tf.keras.activations.sigmoid(x[6+i])
            return tf.stack(x, axis=-1)
        self.activation = tf.keras.layers.Lambda(activation)

    def update(self, sigma, px_size):
        self.cs_layer.sigma = sigma
        #self.cs_layer.px_size = px_size todo: not yet needed

    def __call__(self, inputs):

        x = self.cs_layer(inputs)
        x = self.reshape(x)#72x72
        x = self.batch_norm(x)
        x = self.conv1(x)
        #x = self.pooling_2(x)#36x36
        x = self.conv2(x)
        #x = self.pooling_2(x)#18x18
        #x = self.batch_norm2(x)
        x = self.conv3(x)
        #x = self.pooling_2(x)#9x9
        x = tf.keras.layers.concatenate([x,inputs],axis=-1)
        #x = self.dropout(x)
        x = self.conv4(x)
        #x = self.pooling_3(x)
        #x = self.batch_norm3(x)
        x = tf.keras.layers.Flatten()(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        #todo: sigmoid activation for dense(6:)
        x = self.activation(x)


        return x

    def compute_loss(self, truth_p, predict_p, truth_c, predict_c):
        l2 = tf.keras.losses.MeanSquaredError()
        ce = tf.keras.losses.BinaryCrossentropy()
        RMSE = l2(truth_p, predict_p)#*tf.repeat(truth_c, repeats=[2,2,2],axis=1))
        BCE = tf.reduce_sum(ce(truth_c, predict_c,))
        return 3*RMSE+BCE

    def sort(self, tensor):
        x = tf.unstack(tensor, axis=-1)
        squ = []
        for i in range(len(x)//2):
            i*=2
            squ.append(x[i]**2+x[i+1]**2)
        new = tf.stack(squ, axis=-1)
        return tf.argsort(new, axis=-1, direction='ASCENDING', stable=False, name=None)

    def permute_tensor_structure(self, tensor, indices):
        c = indices+6
        x = indices*2
        y = indices*2+1
        v = tf.stack([x,y],axis=-1)
        v = tf.reshape(v, [indices.shape[0],-1])
        perm = tf.concat([v,c],axis=-1)
        return tf.gather(tensor, perm, batch_dims=1, name=None, axis=-1)

    def compute_permute_loss(self, truth, predict):
        l2 = tf.keras.losses.MeanSquaredError()
        ce = tf.keras.losses.BinaryCrossentropy()

        indices = self.sort(predict[:,0:6])
        indices2 = self.sort(truth[:,0:6])
        predict = self.permute_tensor_structure(predict, indices)
        truth = self.permute_tensor_structure(truth, indices2)
        mask = tf.repeat(truth[:,6:], repeats=[2,2,2],axis=1)
        L2 = l2(predict[:,0:6]*mask, truth[:,0:6])
        BCE = tf.reduce_sum(ce(truth[:,6:], predict[:,6:],))
        return 3*L2+BCE
        #todo: test this

    def compute_alternative_permute_loss(self, truth, predict):
        #todo: compute a per tensor and classifier
        for i in range(3):
            tensor = tf.gather(predict, [2*i,2*i+1,i+6])
            for j in range(3):
                truth = tf.gather(truth, [2*i,2*i+1,i+6])
        #todo: compute for least loss
