import tensorflow as tf
from data import *
import matplotlib.pyplot as plt
from model import *

image = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\coordinate_recon_flim.tif"
cords = r"C:\Users\acecross\PycharmProjects\Wavelet\test_data\coordinate_reconstruction.npz"

gen = data_generator(image, cords)
image = np.zeros((100,64,64))
for i in range(100):
    image[i],cords, ground_truth = gen.__next__()

image_tf1 = tf.convert_to_tensor(image[0:50,:,:,np.newaxis])
image_tf2 = tf.convert_to_tensor(image[50:100,:,:,np.newaxis])
test = model()
test.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

i = image_tf1[0:1]

layer = FullWavelet()
print(layer.weights)
#out = layer(i)


out = test.predict(i)
plt.imshow(out[0,:,:,0])
plt.show()
#test.fit(image_tf1, image_tf2, epochs=5)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
for i in range(STEPS):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0],
                                                       y_: batch[1],
                                                       keep_prob: 1.0})
        print("step {}, training accuracy {}".format(i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],
                                        keep_prob: 0.5})
        X = mnist.test.images.reshape(10, 1000, 784)
        Y = mnist.test.labels.reshape(10, 1000, 10)
        test_accuracy = np.mean([sess.run(accuracy,
                                          feed_dict={x:X[i], y_:Y[i],keep_prob:1.0})
                                 for i in range(10)])
        print("test accuracy: {}".format(test_accuracy))

#todo: images to tensors
#todo: segment images add coordinates
#todo: convolutional NN to coordinates
#todo: compare to localisations
#todo: if localisation is not in rect false negative +=1
#todo: if localisation is in rect true positive += 1
#todo: AI containing parameters for wavelet?

