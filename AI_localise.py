import matplotlib.pyplot as plt
from src.model import *
from src.custom_layers import *

image = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\coordinate_recon_flim.tif"
truth = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\coordinate_reconstruction.npz"



gen = data_generator_coords(image, truth)
image = np.zeros((100,64,64,3))
truth = np.zeros((100,64,64,3))

for i in range(100):
    image[i],truth[i],_ = gen.__next__()

image_tf1 = tf.convert_to_tensor(image[0:50, :, :])
image_tf2 = tf.convert_to_tensor(image[50:100, :, :])
truth_tf1 = tf.convert_to_tensor(truth[0:50, :, :])
truth_tf2 = tf.convert_to_tensor(truth[50:100, :, :])

latent_dim = 2
num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])

epochs = 100
optimizer = tf.keras.optimizers.Adam()
ai = Generator()
# for epoch in range(1, epochs + 1):
#     #for train_x in image_tf1:
#     print(epoch)
#     loss = train_step(ai, image_tf1, optimizer)
#     print(loss)
ai.compile(optimizer='adam',
            loss=generator_loss,
            metrics=['accuracy'])
ai.fit(image_tf1, truth_tf1[:,:,:,:], epochs=1000, validation_data=(image_tf2, truth_tf2))


# mean, logvar = ai.encode(image_tf2[3:4])
# z = ai.reparameterize(mean, logvar)
i = ai.predict(image_tf2[3:4])

fig, axs = plt.subplots(2,4)

axs[0][0].imshow(image_tf2[3,:,:,1])
axs[0][1].imshow(truth_tf2[3,:,:,0])
axs[0][2].imshow(truth_tf2[3,:,:,1])
axs[0][3].imshow(truth_tf2[3,:,:,2])

axs[1][0].imshow(i[0,:,:,0])
axs[1][1].imshow(i[0,:,:,1])
axs[1][2].imshow(i[0,:,:,2])

plt.show()