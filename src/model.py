import tensorflow as tf
from .custom_layers import *

OUTPUT_CHANNELS = 3

def ShiftNet():
    inputs = tf.keras.layers.Input(shape=[9,9,3])
    x=inputs
    conv_stack = [tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)]
    for conv in conv_stack:
        x = conv(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def CompressedSensingNet(cs_layer):
    inputs = tf.keras.layers.Input(shape=[9,9,3])
    x = cs_layer(inputs)#72x72
    #todo: estimate maximum from here?
    conv_stack = [
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3)
    ]
    for conv in conv_stack:
        x = conv(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def ConvNet():
    #todo: adjust input size
    inputs = tf.keras.layers.Input(shape=[9,9,3])
    conv_stack = [
        #downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        # downsample(64, 4, apply_batchnorm=False)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(9, 9, 3), padding="same"),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(2)
    ]
    x=inputs
    for conv in conv_stack:
        x = conv(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return gradients

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    x = tf.cast(x, tf.float32)
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x[:,:,:,1:2])
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),#32
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),#16
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu'),#8
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=8 * 8 * 64, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(8, 8, 64)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=2, padding='same'),  # 8
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ])

    @tf.function
    def sample(self, eps):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def Generator():
    inputs = tf.keras.layers.Input(shape=[64,64,3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)

        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)

        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    activ = tf.keras.layers.ReLU()
    x = last(x)
    x = activ(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

#todo: decoder encoder



LAMBDA = 1
def generator_loss(gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    kl = tf.keras.losses.KLDivergence()
    #sparsity_loss = tf.reduce_sum(tf.abs(gen_output))
    gan_loss = loss_object(gen_output[:,:,:,0:1], target[:,:,:,0:1])
    second_part = tf.multiply(gen_output[:,:,:,0:1], gen_output[:,:,:,1:2])
    third_part = tf.multiply(gen_output[:,:,:,0:1], gen_output[:,:,:,2:3])

    loss = tf.keras.losses.CategoricalCrossentropy()
    CCE = loss(gen_output, target)
    mse = loss(second_part, target[:,:,:,1:2])
    mse += loss(third_part, target[:,:,:,2:3])

    print(mse)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(second_part- target[:,:,:,1:3]))
    secondary_loss = loss_object(gen_output[:,:,:,1:3], target[:,:,:,1:3])

    total_gen_loss = gan_loss + mse#+ (LAMBDA * l1_loss)#gan_loss + (LAMBDA * l1_loss) #+ sparsity_loss

    return total_gen_loss


def wavelet_ai():
    inputs = tf.keras.layers.Input(shape=[128, 128, 1])#todo: input 3 output 1
    layer = FullWavelet(128,level=4,)
    final = tf.keras.layers.ReLU()
    x = inputs
    x = layer(x)
    x = final(x)
    #initializer = tf.random_normal_initializer(0., 0.02)
    #x = tf.math.reduce_sum(x, axis=-1, keepdims=True)
    return tf.keras.Model(inputs=inputs, outputs=x)
