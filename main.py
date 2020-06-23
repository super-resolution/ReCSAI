import tensorflow as tf
from data import *
import matplotlib.pyplot as plt
from model import *
from custom_layers import *
from tensorflow.keras.losses import categorical_crossentropy
import tfwavelets
from tensorflow.keras.optimizers import Adam

image = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\coordinate_recon_flim.tif"
truth = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\coordiante_recon_truth_n_bigtiff.tif"

gen = data_generator_image(image, truth)
image = np.zeros((100,64,64))
truth = np.zeros((100,64,64))

for i in range(100):
    image[i],truth[i] = gen.__next__()

image_tf1 = tf.convert_to_tensor(image[0:50, :, :, np.newaxis])
image_tf2 = tf.convert_to_tensor(image[50:100, :, :, np.newaxis])
truth_tf1 = tf.convert_to_tensor(truth[0:50, :, :, np.newaxis])
truth_tf2 = tf.convert_to_tensor(truth[50:100, :, :, np.newaxis])

#out_o = tfwavelets.nodes.dwt2d(image_tf1[0], tfwavelets.dwtcoeffs.haar)


test = model()
test.compile(optimizer='adam',
            loss=tf.keras.losses.MSE,
            metrics=['accuracy'])

fig, axs = plt.subplots(2,5)
#
# coeffs2 = pywt.dwt2(image[0], 'haar')
# LL, (LH, HL, HH) = coeffs2
#
layer = DWT2()
layer2 = IDWT2()
# print(image_tf1[0:1])
out = layer(image_tf1[0:1])
out = layer2(out)
axs[0,0].imshow(out[0,:,:,0])
axs[1,0].imshow(image_tf1[0,:,:,0])


plt.show()
history = test.fit(image_tf1, truth_tf1[:,:,:,:], epochs=5000, validation_data=(image_tf2, truth_tf2))

i = test.predict(image_tf2[0:1])

fig, axs = plt.subplots(3)

axs[0].imshow(image_tf2[0,:,:,0])
axs[1].imshow(i[0,:,:,0])
axs[2].imshow(truth_tf2[0,:,:,0])
plt.show()

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

test_loss, test_acc = test.evaluate(image_tf2,  truth_tf2, verbose=2)
print(test_acc)


i = image_tf1[0:1]

layer = FullWavelet()
print(layer.weights)
out = layer(i)

fig, axs = plt.subplots(1,5)
axs[0].imshow(i[0,:,:,0])
#out = test.predict(i)
for i in range(out.shape[-1]):
    j= i+1
    axs[j].imshow(out[0,:,:,i])
plt.show()
#test.fit(image_tf1, image_tf2, epochs=5)


