from src.model import *
from src.data import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.visualization import display_storm_data
import pandas as pd
from result_evaluation import jaccard_index
#import result_evaluation
#import tensorboard
from tifffile import TiffWriter



#done: load wavelet checkpoints
#denoising = wavelet_ai()

#checkpoint_path = "training_lvl2/cp-10000.ckpt"

#denoising.load_weights(checkpoint_path)




#cs_net = CompressedSensingNet(CompressedSensing())
#checkpoint_path = "cs_training/cp-0010.ckpt" #todo: load latest checkpoint
# Create a callback that saves the model's weights every 5 epochs
#cs_net.load_weights(checkpoint_path)
#print(cs_net.weights)

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


def predict_localizations(path):
    drift = pd.read_csv(r"C:\Users\biophys\Downloads\confocalSTORM_beads+MT\alpha_drift.csv").as_matrix()
    result_array = []
    gen = generate_generator(path)
    dataset = tf.data.Dataset.from_generator(gen, tf.float64)
    j = 0
    for image in dataset:
        # plt.imshow(image[2,:,:,0])
        # plt.show()
        # plt.imshow(image[2,:,:,1])
        # plt.show()
        crop_tensor, _, coord_list = bin_localisations_v2(image, denoising, th=0.35)
        for z in range(len(coord_list)):
            coord_list[z][2] += j*5000
        print(crop_tensor.shape[0])
        result_tensor = cs_net.predict(crop_tensor)
        # todo: extrude psf without drift
        checkpoint_path = "psf_training/cp-{epoch:04d}.ckpt"

        latest = tf.train.latest_checkpoint(
            "psf_training", latest_filename=None
        )
        sigma_predict = ParamNet()

        # Create a callback that saves the model's weights every 5 epochs
        sigma_predict.load_weights(latest)
        predict_sigma(crop_tensor, result_tensor, sigma_predict)
        #
        # #psf = extrude_perfect_psf(crop_tensor, result_tensor)
        #
        # new_layer = CompressedSensing()
        # new_layer.update_psf(psf/2)
        # cs_net_updated = CompressedSensingNet(new_layer)
        # checkpoint_path = "cs_training3/cp-0010.ckpt"  # todo: load latest checkpoint
        # # Create a callback that saves the model's weights every 5 epochs
        # cs_net_updated.load_weights(checkpoint_path)
        # # cs_net_updated.compile(optimizer='adam',
        # #                loss=tf.keras.losses.MSE,
        # #                metrics=['accuracy'])
        # result_tensor = cs_net_updated.predict(crop_tensor)
        fig, axs = plt.subplots(3)
        axs[0].imshow(crop_tensor[100,:,:,0])
        axs[0].scatter(result_tensor[100,1]/8,result_tensor[100,0]/8)
        axs[1].imshow(crop_tensor[100,:,:,1])
        axs[1].scatter(result_tensor[100,1]/8,result_tensor[100,0]/8)
        axs[2].imshow(crop_tensor[100,:,:,2])
        axs[2].scatter(result_tensor[100,1]/8,result_tensor[100,0]/8)
        plt.show()
        # del crop_tensor
        frame=0
        for i in range(result_tensor.shape[0]):
            if result_tensor[i,2]>0.6 :
                current_drift = drift[int(coord_list[i][2]*0.4),1:3]
                #current_drift[1] *= -1
#                if coord_list[i][2] == frame:
                result_array.append(coord_list[i][0:2] + np.array([result_tensor[i,0]/8, result_tensor[i,1]/8]))
                # else:
                #     frame +=1
                #     result_array = np.array(result_array)
                #     plt.scatter(result_array[:,0],result_array[:,1])
                #     plt.show()
                #     result_array = []

        del result_tensor
        j+=1
    result_array = np.array(result_array)
    #result_array[:,0] += 45
    print(result_array.shape[0])

    print("finished AI")
    display_storm_data(result_array)
    np.save(os.getcwd()+r"\DNApaint.npy",result_array)

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


def train_cs_net():
    #test dataset from generator
    for i in range(10):
        # generator = real_data_generator(100)
        # gen = generator()
        # #dataset = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32, tf.float64))
        # image = np.zeros((1000, 128, 128))
        # #todo: use crop generator
        # truth_coords = []
        # #im,truth,locs = dataset.batch(10)
        # for i in range(1000):#todo: 1000 images learn redo...
        #     image[i],_,loc = gen.__next__()
        #     truth_coords.append(loc)
        # # with TiffWriter(os.getcwd() + r"\dataset3.tif",
        # #                 bigtiff=True) as tif:
        # #     tif.save(image[:,14:-14,14:-14], photometric='minisblack')
        # truth_coords = np.array(truth_coords)
        # truth_coords /= 100
        # truth_coords +=14.5
        # data = np.zeros((image.shape[0], 128, 128, 3))  # todo: this is weird data
        # data[:, :image.shape[1], : image.shape[2], 1] = image
        # data[1:, : image.shape[1], : image.shape[2], 0] = image[:-1]
        # data[:-1, : image.shape[1], : image.shape[2], 2] = image[1:]
        # image = data

        # image = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction_flim.tif"
        # truth = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\100x100maxi_batch\coordinate_reconstruction.npz"
        # gen = data_generator_coords(image, offset=0)
        # image = np.zeros((1000, 128, 128, 3))
        # #truth = np.zeros((100, 64, 64, 3))
        # truth_coords = np.load(truth, allow_pickle=True)['arr_0']
        # truth_coords /= 100
        # truth_coords += 14.5#thats offset
        # for i in range(1000):
        #     image[i] = gen.__next__()
        #    fig,axs = plt.subplots(3)
        #     axs[0].imshow(image[i,:,:,0])
        #     axs[1].imshow(image[i,:,:,1])
        #     axs[2].imshow(image[i,:,:,2])
        #     plt.show()
        # del gen

        # for value in dataset:
        #     print(value[1][0], value[2][0])
        #     fig,axs = plt.subplots(3)
        #     axs[0].imshow(value[0][0,:,:,0])
        #     axs[1].imshow(value[0][0,:,:,1])
        #     axs[2].imshow(value[0][0,:,:,2])
        #     plt.show()
        cs_net = CompressedSensingNet()

        #checkpoint_path = "cs_training/cp-{epoch:04d}.ckpt"  # done: load latest checkpoint
        optimizer = tf.keras.optimizers.Adam()
        step = tf.Variable(1, name="global_step")
        #accuracy = tf.metrics.Accuracy()
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=cs_net)
        manager = tf.train.CheckpointManager(ckpt, './cs_training', max_to_keep=3)


        @tf.function
        def train_step(train_image, truth):
            with tf.GradientTape() as tape:
                truth_p = truth[:,0:6]/100
                logits = cs_net(train_image)
                logits_p = logits[:,0:6]/8
                loss = compute_cs_loss(truth_p, logits_p, truth[:,6:], logits[:,6:], )
            gradients = tape.gradient(loss, cs_net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, cs_net.trainable_variables))
            #step.assign_add(1)

            #accuracy_value = accuracy(truth, tf.argmax(logits, -1))
            return loss#, accuracy_value

        #@tf.function
        def loop(dataset):
            for train_image, sigma, truth in dataset.take(15):
                for i in range(50):
                    loss_value = train_step(train_image, truth)
                    ckpt.step.assign_add(1)
                    if int(ckpt.step) % 10 == 0:
                        save_path = manager.save()
                        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                        print("loss {:1.2f}".format(loss_value.numpy()))

        def outer_loop():
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")
            for j in range(5):
                sigma = np.random.randint(100, 250)
                generator = crop_generator(9, sigma_x=sigma)
                dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32), output_shapes=((100,9,9,3),(),(100,9)))
                cs_net.update(sigma, 100)
                loop(dataset)
            test(dataset)

        def test(dataset):
            for train_image, sigma, truth in dataset.take(20):
                truth = truth.numpy() / 100
                result = cs_net.predict(train_image) / 8
                for i in range(truth.shape[0]):
                    plt.imshow(train_image[i, :, :, 1])
                    for n in range(3):
                        plt.scatter(truth[i,2*n+1],truth[i,2*n], c="r")
                        if result[i,6+n]<0.5:
                            plt.scatter(result[i,2*n+1],result[i,2*n], c="g")
                    plt.show()


        outer_loop()
        #image_tf1 = tf.convert_to_tensor(image[0:900, :, :])
        #image_tf2 = tf.convert_to_tensor(image[900:1000, :, :])#todo: use 20% test
        #truth_tf1 = tf.convert_to_tensor(truth[0:90, :, :])
        #truth_tf2 = tf.convert_to_tensor(truth[90:100, :, :])#todo: use 20% test

        #train_new, truth_new,_ = bin_localisations_v2(image_tf1, denoising, truth_array=truth_coords[:], th=0.2)
        #logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_path,
        #     verbose=1,
        #     save_weights_only=True,
        #     period=10)
        # cs_net.compile(optimizer='adam',
        #              loss=tf.keras.losses.MSE,
        #              metrics=['accuracy'])
        # test_new, truth_test_new, test_coord_list = bin_localisations_v2(image_tf2, denoising, truth_array=truth_coords[900:], th=0.25)#todo: train on localisations directly
        #
        # cs_net.fit(train_new, truth_new, epochs=50, callbacks=[cp_callback],validation_data=[test_new, truth_test_new] )

    result_array = cs_net.predict(test_new)


    # psf = extrude_perfect_psf(test_new, result_array)
    # #todo: set psf in model
    # new_layer = CompressedSensing()
    # new_layer.update_psf(psf/2)
    # cs_net_updated = CompressedSensingNet(new_layer)
    # checkpoint_path = "cs_training3/cp-0010.ckpt"  # todo: load latest checkpoint
    # # Create a callback that saves the model's weights every 5 epochs
    # cs_net_updated.load_weights(checkpoint_path)
    # # cs_net_updated.compile(optimizer='adam',
    # #                loss=tf.keras.losses.MSE,
    # #                metrics=['accuracy'])
    # result_array = cs_net_updated.predict(test_new)

    per_fram_locs = []
    current_frame = 0
    current_frame_locs = []
    for i in range(result_array.shape[0]):
        r = test_coord_list[i][0:2] + result_array[i,0:2]/8
        if test_coord_list[i][2] == current_frame and result_array[i,2]>0.9:
            current_frame_locs.append(r)
        elif result_array[i,2]>0.9:
            per_fram_locs.append(np.array(current_frame_locs))
            current_frame = test_coord_list[i][2]
            current_frame_locs = []
            current_frame_locs.append(r)
    #append last frame
    per_fram_locs.append(np.array(current_frame_locs))
    #todo: create a test function for jaccard
    per_fram_locs = np.array(per_fram_locs)*100
    current_truth_coords = truth_coords[900:1000]
    current_truth_coords *=100

    result, false_positive, false_negative, jac, rmse = jaccard_index(per_fram_locs, current_truth_coords)
    false_negative = false_negative[0]
    false_positive = false_positive[0]
    print(false_negative.shape[0], false_positive.shape[0])
    dat = np.mean(result[:,2:4],axis=0)
    for i in range(current_truth_coords.shape[0]):
        fp = np.array(false_positive[i])
        fn = np.array(false_negative[i])
        fig,axs = plt.subplots(3)
        axs[0].scatter(current_truth_coords[i][:,0],current_truth_coords[i][:,1])
        axs[0].scatter(per_fram_locs[i][:,0], per_fram_locs[i][:,1])
        axs[1].scatter(fp[:,0],fp[:,1])
        axs[2].scatter(fn[:,0],fn[:,1])
        #plt.scatter(dat[0],dat[1])
        plt.show()
    for i in range(test_new.shape[0]):
        plt.imshow(test_new[i,:,:,1])

        plt.scatter(truth_test_new[i:i+1, 1]/8, truth_test_new[i:i+1, 0]/8, c="r")
        plt.scatter(result_array[i:i+1, 1]/8, result_array[i:i+1, 0]/8)
        print(result_array[:, 2])

        plt.show()

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


#validate_cs_model()
train_cs_net()
#train_nonlinear_shifter_ai()
#learn_psf()



image = r"D:\Daten\Dominik_B\Cy5_MT_100us_101nm_45px_Framesfrq2.4Hz_Linefrq108.7Hz_5kW_7500Frames_kept stack.tif"
#image = r"D:\Daten\AI\COS7_LT1_beta-tub-Alexa647_new_D2O+MEA5mM_power6_OD0p6_3_crop.tif"
#image = r"C:\Users\biophys\matlab\test2_crop_BP.tif"
#image = r"D:\Daten\Artificial\ContestHD.tif"
#image = r"D:\Daten\Domi\origami\201203_10nM-Trolox_ScSystem_50mM-MgCl2_kA_TiRF_568nm_100ms_45min_no-gain-10MHz_zirk.tif"
predict_localizations(image)


#done: binnin here
#fig,ax = plt.subplots(1)
image = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\maxi_batch\coordinate_reconstruction_flim.tif"
truth = r"C:\Users\biophys\PycharmProjects\ISTA\artificial_data\maxi_batch\coordinate_reconstruction.npz"
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