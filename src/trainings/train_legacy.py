
def train_recon_net():
    image = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\maxi_batch\coordinate_reconstruction_flim.tif"
    truth = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\maxi_batch\coordinate_reconstruction.npz"
    gen = data_generator_coords(image, truth)
    image = np.zeros((16000, 64, 64, 3))
    #truth = np.zeros((100, 64, 64, 3))
    checkpoint_path = "recon_training/cp-{epoch:04d}.ckpt"  # done: load latest checkpoint
    truth_coords = np.load(truth, allow_pickle=True)['arr_0']
    truth_coords[:,:,0:2] /= 100
    truth_coords[:,:,0:2] +=8
    for i in range(16000):
        image[i] = gen.__next__()
    #    fig,axs = plt.subplots(3)
    #     axs[0].imshow(image[i,:,:,0])
    #     axs[1].imshow(image[i,:,:,1])
    #     axs[2].imshow(image[i,:,:,2])
    #     plt.show()
    # del gen


    image_tf1 = tf.convert_to_tensor(image[0:15000, :, :])
    image_tf2 = tf.convert_to_tensor(image[15000:16000, :, :])#todo: use 20% test
    #truth_tf1 = tf.convert_to_tensor(truth[0:90, :, :])
    #truth_tf2 = tf.convert_to_tensor(truth[90:100, :, :])#todo: use 20% test

    train_new, truth_new,_ = bin_localisations_v2(image_tf1, denoising, truth_array=truth_coords)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=20)
    recon_net.compile(optimizer='adam',
                 loss=tf.keras.losses.MSE,
                 metrics=['accuracy'])
    test_new, truth_test_new,_ = bin_localisations_v2(image_tf2, denoising, truth_array=truth_coords[15000:16000])#todo: train on localisations directly


    recon_net.fit(train_new, truth_new, epochs=100, callbacks=[cp_callback],validation_data=[test_new, truth_test_new] )

    result_array = recon_net.predict(test_new[1:2])
    plt.imshow(test_new[1,:,:,1])

    plt.scatter(truth_test_new[1, 1], truth_test_new[1, 0])
    plt.scatter(result_array[:, 1], result_array[:, 0])

    plt.show()


def validate_cs_model():
    image = os.getcwd() + r"\test_data\dataset9.tif"#r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction_flim.tif"
    truth = os.getcwd() + r"\test_data\dataset9.npy"#r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction.npz"
    gen = data_generator_coords(image, offset=0)
    image = np.zeros((100, 128, 128, 3))
    #truth = np.zeros((100, 64, 64, 3))
    truth_coords = np.load(truth, allow_pickle=True)
    truth_coords /= 100
    truth_coords += 14.5#thats offset
    for i in range(100):
        image[i] = gen.__next__()

    image_tf1 = tf.convert_to_tensor(image[0:100, :, :])
    #image_tf2 = tf.convert_to_tensor(image[90:100, :, :])#todo: use 20% test
    #truth_tf1 = tf.convert_to_tensor(truth[0:90, :, :])
    #truth_tf2 = tf.convert_to_tensor(truth[90:100, :, :])#todo: use 20% test

    data, truth_new, coord_list = bin_localisations_v2(image_tf1, denoising, truth_array=truth_coords[:], th=0.25)
    result_array = cs_net.predict(data)
    per_fram_locs = []
    current_frame = 0
    current_frame_locs = []
    per_frame_multifit = []
    multifit = []
    for i in range(result_array.shape[0]):
        r = coord_list[i][0:2] + result_array[i,0:2]/8
        if coord_list[i][2] == current_frame and result_array[i,2]>0.5:
            if result_array[i,2]<1.4:
                current_frame_locs.append(r)
            else:
                multifit.append(r*100)
        elif result_array[i,2]>0.5:
            per_fram_locs.append(np.array(current_frame_locs))
            per_frame_multifit.append(np.array(multifit))
            current_frame = coord_list[i][2]
            current_frame_locs = []
            multifit = []
            if result_array[i,2]<1.4:
                current_frame_locs.append(r)
            else:
                multifit.append(r*100)
    #append last frame
    per_frame_multifit.append(np.array(multifit))
    per_fram_locs.append(np.array(current_frame_locs))
    #todo: create a test function for jaccard
    #per_frame_multifit = np.array(multifit)*100
    per_fram_locs = np.array(per_fram_locs)*100
    current_truth_coords = truth_coords[:100]
    current_truth_coords *=100

    result, false_positive, false_negative, jac, rmse = jaccard_index(per_fram_locs, current_truth_coords)
    false_positive = false_positive[0]
    false_negative = false_negative[0]
    print(false_negative.shape[0], false_positive.shape[0])
    for i in range(current_truth_coords.shape[0]):
        fig,axs = plt.subplots(3)
        if len(false_positive[i])>0:
            fp = np.array(false_positive[i])
            axs[1].scatter(fp[:, 0], fp[:, 1])
        if false_negative[i].shape[0]>0:
            fn = np.array(false_negative[i])
            axs[2].scatter(fn[:, 0], fn[:, 1])

        axs[1].imshow(image[i,:,:,1])
        axs[0].scatter(current_truth_coords[i][:,0],current_truth_coords[i][:,1])
        if per_fram_locs[i].shape[0] !=0:
            axs[0].scatter(per_fram_locs[i][:,0], per_fram_locs[i][:,1])
        if len(per_frame_multifit[i])!=0:
            axs[0].scatter(per_frame_multifit[i][0][0], per_frame_multifit[i][0][1],c="g")
        #plt.scatter(dat[0],dat[1])
        plt.show()

def learn_psf():
    ai = ParamNet()

    for i in range(4):
        crops = np.zeros((1001, 9, 9, 10))
        truth = np.zeros((1001,3))

        truth_coords = []
        #im,truth,locs = dataset.batch(10)
        for i in range(1001):#todo: 1000 images learn redo...
            generator = crop_generator(9)()
            for j in range(10):
                crops[i,:,:,j], truth[i] = generator.__next__()
                #print(truth[i])
                crops[i, :, :, j] -= crops[i, :, :, j].min()
                crops[i, :, :, j] /= crops[i, :, :, j].max()
                #
                # plt.imshow(crops[i,:,:,j])
                # plt.show()

        train_data = tf.convert_to_tensor(crops[0:1000])
        truth_data = tf.convert_to_tensor(truth[0:1000]/100)
        checkpoint_path = "psf_training/cp-{epoch:04d}.ckpt"

        latest = tf.train.latest_checkpoint(
            "psf_training", latest_filename=None
        )

        # Create a callback that saves the model's weights every 5 epochs
        #ai.load_weights(latest)


        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=100)


        ai.compile(optimizer='adam',
                     loss=tf.keras.losses.MSE,
                     metrics=['accuracy'])
        ai.fit(train_data, truth_data, callbacks=[cp_callback], epochs=100)
        test_data = tf.convert_to_tensor(crops[1000:1001])
        test_truth_data = tf.convert_to_tensor(truth[1000:1001])
        x = ai.predict(test_data)
        print(x, test_truth_data/100)

def train_nonlinear_shifter_ai():
    image = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100mini_batch\coordinate_reconstruction_flim.tif"
    truth = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100mini_batch\coordinate_reconstruction.npz"
    gen = data_generator_coords(image)
    #gen_t = data_generator_coords(truth)

    truth_coords = np.load(truth, allow_pickle=True)['arr_0']
    for i in range(truth_coords.shape[0]):
        truth_coords[i][:,0:2] /= 100
        truth_coords[i][:,0:2] += 8#thats offset

    shift_ai = ShiftNet()

    shift_ai.compile(optimizer='adam',
                 loss=tf.keras.losses.MSE,
                 metrics=['accuracy'])

    image = np.zeros((1000, 128, 128, 3))
    #truth = np.zeros((100, 128, 128, 3))

    for i in range(1000):
        image[i] = gen.__next__()
        #truth[i] = gen_t.__next__()


    image_tf1 = tf.convert_to_tensor(image[0:900, :, :])
    image_tf2 = tf.convert_to_tensor(image[900:1000, :, :])#todo: use 20% test
    #truth_tf1 = tf.convert_to_tensor(truth_coords[0:90, :, :])
    #truth_tf2 = tf.convert_to_tensor(truth_coords[90:100, :, :])#todo: use 20% test



    checkpoint_path = "shift_training/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=100)

    train_new, truth_new,_ = create_shift_data(image_tf1, denoising, truth_array=truth_coords, th=0.15)

    test_new, truth_test_new,_ = create_shift_data(image_tf2, denoising, truth_array=truth_coords[900:], th=0.15)

    truth_new = truth_new[:,4]/100
    truth_test_new = truth_test_new[:,4]/100
    shift_ai.fit(train_new, truth_new, epochs=1000, callbacks=[cp_callback], validation_data=[test_new, truth_test_new] )

    for i in range(test_new.shape[0]):
        result = shift_ai.predict(test_new[i:i+1])

        layer = Shifting()
        result_crop = layer(test_new[i:i+1],tf.stack([result[0,0],0.0]))

        fig,axs = plt.subplots(2)
        axs[0].imshow(test_new[i,:,:,1])
        axs[1].imshow(result_crop[0,:,:,1])
        #axs[2].imshow(truth_test_new[1,:,:,1])
        plt.show()