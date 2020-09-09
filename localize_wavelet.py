import os
from src.model import *
from src.data import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.utility import bin_localisations
from visualization import display_storm_data




#done: load wavelet checkpoints
denoising = wavelet_ai()

checkpoint_path = "training_lvl2/cp-1000.ckpt"

denoising.load_weights(checkpoint_path)



recon_net = ConvNet()
checkpoint_path = "recon_training/cp-0500.ckpt" #todo: load latest checkpoint
# Create a callback that saves the model's weights every 5 epochs
recon_net.load_weights(checkpoint_path)

def train_recon_net():
    image = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\maxi_batch\coordinate_reconstruction_flim.tif"
    truth = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\maxi_batch\coordinate_reconstruction.npz"
    gen = data_generator_coords(image, truth)
    image = np.zeros((1000, 64, 64, 3))
    truth = np.zeros((1000, 64, 64, 3))
    checkpoint_path = "recon_training/cp-{epoch:04d}.ckpt"  # done: load latest checkpoint

    for i in range(1000):
        image[i], truth[i],_ = gen.__next__()
        fig,axs = plt.subplots(3)
        axs[0].imshow(image[i,:,:,0])
        axs[1].imshow(image[i,:,:,1])
        axs[2].imshow(image[i,:,:,2])
        plt.show()



    image_tf1 = tf.convert_to_tensor(image[0:900, :, :])
    image_tf2 = tf.convert_to_tensor(image[900:1000, :, :])#todo: use 20% test
    truth_tf1 = tf.convert_to_tensor(truth[0:900, :, :])
    truth_tf2 = tf.convert_to_tensor(truth[900:1000, :, :])#todo: use 20% test

    train_new, truth_new,_ = bin_localisations(image_tf1, denoising, truth_tensor=truth_tf1)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=50)
    recon_net.compile(optimizer='adam',
                 loss=tf.keras.losses.MSE,
                 metrics=['accuracy'])
    test_new, truth_test_new,_ = bin_localisations(image_tf2, denoising, truth_tensor=truth_tf2)#todo: train on localisations directly

    recon_net.fit(train_new, truth_new, epochs=500, callbacks=[cp_callback],validation_data=[test_new, truth_test_new] )


def predict_localizations(data_tensor):
    result_array = []

    crop_tensor, _, coord_list = bin_localisations(data_tensor, denoising, th=35.0)

    result_tensor = recon_net.predict(crop_tensor)
    for i in range(result_tensor.shape[0]):
        result_array.append(coord_list[i]+np.array([result_tensor[i,0],result_tensor[i,1]]))
    result_array = np.array(result_array)
    display_storm_data(result_array)
    np.save(os.getcwd()+r"\test.npy",result_array)

#train_recon_net()

image = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\real_data\Cy5.tif"
image = data_generator_real(image)
image_tf = tf.convert_to_tensor(image)
predict_localizations(image_tf)


#done: binnin here
#fig,ax = plt.subplots(1)
image = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\maxi_batch\coordinate_reconstruction_flim.tif"
truth = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\maxi_batch\coordinate_reconstruction.npz"
bin = Binning()
gen = data_generator_coords(image, truth, offset=900)
image = np.zeros((50, 64, 64, 3))
truth = np.zeros((50, 64, 64, 3))

for i in range(50):
    image[i], truth[i], _ = gen.__next__()

    # todo: wavelet prediciton here

#todo: this is plotting and testing
image_tf2 = tf.convert_to_tensor(image[:50, :, :])
truth_tf2 = tf.convert_to_tensor(truth[:50, :, :])
test_new = []
truth_test = []
result_array = []
i=7
one = denoising.predict(image_tf2[i:i + 1, :, :, 0:1])
two = denoising.predict(image_tf2[i:i + 1, :, :, 1:2])
three = denoising.predict(image_tf2[i:i + 1, :, :, 2:3])
im = tf.concat([one, two, three], -1)

#wave = image_tf1[i:i + 1, :, :, 1:2]
wave = denoising.predict(image_tf2[i:i+1,:,:,1:2])
y = tf.constant([8])
mask = tf.greater(wave,y)
wave_masked = wave*tf.cast(mask,tf.float32)
#ax.imshow(wave[0,:,:,0])
coords = bin.get_coords(wave_masked.numpy()[0,:,:,0])
# todo: crop PSFs done here
plt.imshow(image_tf2[i, :, :, 1])
plt.show()
#plt.imshow(im[0, :, :, 1:2])
#plt.show()
fig, ax = plt.subplots(1)

for coord in coords:
    rect = patches.Rectangle((coord[1] - 3, coord[0] - 3), 6, 6, linewidth=0.5, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    # ax.set_title("original", fontsize=10)
    if coord[0]-4>0 and coord[1]-4>0and coord[0]+4<64 and coord[1]+4<64:
        crop = im[0,coord[0]-4:coord[0]+5,coord[1]-4:coord[1]+5, :]
        np.save(os.getcwd() + r"\crop.npy", crop)

        #crop_wave = wave[0,coord[0]-4:coord[0]+5,coord[1]-4:coord[1]+5, :]
        #crop = tf.concat([crop, crop_wave], -1)
        truth = truth_tf2[i:i+1,coord[0]-1:coord[0]+2,coord[1]-1:coord[1]+2, :]
        ind = tf.argmax(tf.reshape(truth[0,:,:,0],[-1]))
        x = ind // truth.shape[2]
        y = ind % truth.shape[2]
        x_f = tf.cast(x, tf.float64)
        y_f = tf.cast(y, tf.float64)
        x_f += truth[0,x,y,1] + 3 #add difference to crop dimension
        y_f += truth[0,x,y,2] + 3
        test_new.append(crop)
        truth_test.append(tf.stack([x_f,y_f]))
        result = recon_net.predict(tf.expand_dims(crop,0))
        current_entry = np.array([[float(result[0,0]),float(result[0,1])],[float(x_f),float(y_f)]])
        current_entry[:,0] += coord[0]-4
        current_entry[:,1] += coord[1]-4
        result_array.append(current_entry)

result_array = np.array(result_array)
ax.imshow(im[0,:, :, 1])

ax.scatter(result_array[:,0,1], result_array[:,0,0])
ax.scatter(result_array[:,1,1], result_array[:,1,0])

plt.show()
#todo: back to coordinates

test_new = tf.stack(test_new)
x = recon_net.predict(test_new)
y=0

#todo: ne+w dataset with psf[i],i-1,i+
#todo: new truth with adjusted values




#todo: learn reconstruction on coordinates and compare to data