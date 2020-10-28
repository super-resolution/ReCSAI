import matplotlib.pyplot as plt
from src.model import *
from src.custom_layers import *
import os

image = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100mini_batch\coordinate_reconstruction_flim.tif"
truth = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100mini_batch\coordinate_reconstruction_truth.tif"

gen = real_data_generator(100)()
image = np.zeros((1000,128,128))
truth = np.zeros((1000,128,128))

for i in range(1000):
    image[i],truth[i],_ = gen.__next__()
    image[i] /= image[i].max()+0.0001
    truth[i] /= truth[i].max()+0.0001

data = np.zeros((image.shape[0], 128, 128, 3))  # todo: this is weird data
data[:, :image.shape[1], : image.shape[2], 1] = image
data[1:, : image.shape[1], : image.shape[2], 0] = image[:-1]
data[:-1, : image.shape[1], : image.shape[2], 2] = image[1:]
image = data

data = np.zeros((truth.shape[0], 128, 128, 3))  # todo: this is weird data
data[:, :truth.shape[1], : truth.shape[2], 1] = truth
data[1:, : truth.shape[1], : truth.shape[2], 0] = truth[:-1]
data[:-1, : truth.shape[1], : truth.shape[2], 2] = truth[1:]
truth = data

image_tf1 = tf.convert_to_tensor(image[0:900, :, :, :])
image_tf2 = tf.convert_to_tensor(image[900:1000, :, :, :])
truth_tf1 = tf.convert_to_tensor(truth[0:900, :, :, :])
truth_tf2 = tf.convert_to_tensor(truth[900:1000, :, :, :])

print("data loaded")
#out_o = tfwavelets.nodes.dwt2d(image_tf1[0], tfwavelets.dwtcoeffs.haar)
checkpoint_path = "training_lvl3/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=50)



test = wavelet_ai()
test.load_weights("training_lvl3/cp-1000.ckpt")

test.compile(optimizer='adam',
            loss=tf.keras.losses.MSE,
            metrics=['accuracy'])

fig, axs = plt.subplots(2,5)
#
# coeffs2 = pywt.dwt2(image[0], 'haar')
# LL, (LH, HL, HH) = coeffs2
#
layer = DWT2()
layer2 = IDWT2(128)
# print(image_tf1[0:1])

out = layer(image_tf1[1:2,:,:,0:1])
out = layer2(out)
#axs[0,0].imshow(out[0,:,:,0])
#axs[1,0].imshow(image_tf1[1,:,:,0])
#plt.show()
print(tf.reduce_sum(test.trainable_weights[1][0]*test.trainable_weights[1][1]))

history = test.fit(image_tf1[:,:,:,1:2], truth_tf1[:,:,:,1:2], epochs=10000, validation_data=(image_tf2[:,:,:,1:2], truth_tf2[:,:,:,1:2]), callbacks=[cp_callback])

i = test.predict(image_tf2[1:2,:,:,1:2])

print(tf.reduce_sum(test.trainable_weights[1][0]*test.trainable_weights[1][1]))

fig, axs = plt.subplots(3)

axs[0].imshow(image_tf2[1,:,:,1])
axs[1].imshow(i[0,:,:,0])
axs[2].imshow(truth_tf2[1,:,:,1])
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = test.evaluate(image_tf2,  truth_tf2, verbose=2)



