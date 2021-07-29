import tensorflow as tf
from src.models.cs_model import CompressedSensingNet, CompressedSensingCVNet, CompressedSensingInceptionNet
from unittest import skip
from tests.test import BaseTest
import copy
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors
import seaborn as sns
import functools
import matplotlib.pyplot as plt; plt.style.use('ggplot')


class TestCompressedSensingNet(BaseTest):
    def setUp(self):
        self.network = CompressedSensingNet()

    def permute_tensor_structure(self, tensor, indices):
        c = indices+6
        x = indices*2
        y = indices*2+1
        v = tf.transpose([x,y])
        v = tf.reshape(v, [indices.shape[0],-1])
        perm = tf.concat([v,c],axis=-1)
        return tf.gather(tensor, perm, batch_dims=1, name=None, axis=-1)


    def test_update_psf(self):
        self.network.cs_layer.sigma = 150
        initial_psf = copy.deepcopy(self.network.cs_layer.mat.numpy())
        self.network.update(180, 100)
        new_psf = self.network.cs_layer.mat.numpy()
        self.assertNotAllClose(initial_psf, new_psf, msg="PSF update doesnt work")

    def test_network_learns_loss_function(self):
        #todo: apply l1 regularizer on cs path?
        #todo: loss function for cs path
        #todo: penaltize zeros and apply dropout in this layer?
        self.fail()

    def test_cs_layer_output_not_zero(self):
        self.fail()

    def test_loss_is_legit(self):
        self.fail()

    def test_permutation_loss(self):
        #todo: test multiple permutations
        tensor1 = tf.constant([[30,20,15,7,0,0,1,1,0],[13,5,0,0,15,10,1,0,1]],dtype=tf.float32)#todo: permute tensor and receive loss of zero?
        tensor2 = self.network.permute_tensor_structure(tensor1, tf.constant([[2,1,0],[0,1,2]]))
        loss = self.network.compute_permute_loss(tensor1, tensor2)
        self.assertEqual(loss,0, msg="loss for only permuted tensors is not zero")
        tensor3 = tf.constant([[29, 22, 15, 7, 0, 0, 1, 1, 0],[13,5,0,0,15,10,1,0,1] ], dtype=tf.float32)
        loss = self.network.compute_permute_loss(tensor3, tensor2)
        self.assertAllClose(loss, (1.0+4.0)/2.0, msg="loss for values is not rmse")

        #todo: permute classifier and tensor together

        #todo: it should be neglectable wether the position is reconstructed in vector 1 2 or 3


class TestCsInceptionNet(BaseTest):
    def setUp(self):
        self.network = CompressedSensingInceptionNet()

    def test_sigma_updates_in_both_cs_layers(self):
        self.network.sigma = 150
        initial_psf = copy.deepcopy(self.network.inception1.cs.mat.numpy())
        self.network.sigma = 180
        new_psf = self.network.inception1.cs.mat.numpy()
        self.assertNotAllClose(initial_psf, new_psf, msg="PSF update doesnt work")
        new_psf2 = self.network.inception2.cs.mat.numpy()
        self.assertAllClose(new_psf, new_psf2)

    def test_decode_loss(self):
        import numpy as np
        x = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])  # [tf.newaxis,tf.newaxis,tf.newaxis,:]
        X, Y = tf.meshgrid(x, x)
        truth = tf.constant([
            [[4.5, 6.3,1], [-1, -1,0], [-1, -1,0]],

            [[2.2,4.5,1],[3,2,1],[-1,-1,0]],
                             ], dtype=tf.float32)
        vec = np.zeros((2,9,9,6))
        vec[1,:,:,2] = 0.001

        vec[1,1,4,2] = 0.9
        vec[1,1,4,0] = -0.3
        vec[1,1,4,1] = 0
        vec[1,:,:,3] = 1
        vec[1,:,:,4] = 1

        vec[1,3,2,2] = 0.9
        vec[1,3,2,0] = -0.5
        vec[1,3,2,1] = -0.5
        vec[1,:,:,3] = 1
        vec[1,:,:,4] = 1

        vec[0,:,:,2] = 0.001

        vec[0,4,6,2] = 0.9
        vec[0,4,6,0] = -0.
        vec[0,4,6,1] = -0.2
        vec[0,:,:,3] = 0.1
        vec[0,:,:,4] = 0.1
        #vec[1,1,2,2]=1

        predict = tf.constant(vec, dtype=tf.float32)
        #todo: set right entries...
        #predict = predict[0]
        #truth = truth[0]
        #z=0
        test = (predict[0,:, :, 2] /
                (tf.reduce_sum(predict[0,:, :, 2])
                 * tf.math.sqrt(tf.math.sqrt((predict[0,:, :, 3])) *
                                #tf.pow(2 * 3.14, 4) *
                                tf.math.sqrt(predict[0,:, :, 4]))
                 ))
        #         * tf.exp(-(1 / 2 * tf.square(
        #             predict[:, :, 0] + Y - truth[0][0])  # todo: test that this gives expected values
        #                    / predict[:, :, 3]
        #                    + tf.square(predict[:, :, 1] + X - truth[0][1])
        #                    / predict[:, :, 4]  # todo: activation >= 0
        #                    ))).numpy()
        i=0
        print(test)
        # print(tf.exp(-(1 / 2 * tf.square(
        #                                 predict[:,:, :, 0] + Y - truth[:,i:i+1,0:1])  # todo: test that this gives expected values
        #                                  / predict[:,:, :, 3]
        #                                  + tf.square(predict[:,:, :, 1] + X - truth[:,i:i+1,1:2])
        #                                  / predict[:, :, :, 4]  # todo: activation >= 0
        #                                  )))
        print(self.network.compute_loss_decode(truth,predict,0))#todo: adjust parameters
        self.fail()



class TestCsUNet(BaseTest):
    def setUp(self):
        self.network = CompressedSensingCVNet()

    def test_shapes(self):
        data = self.create_noiseless_random_data_crop(9, 150, 100)
        y = data
        for layer in self.network.down_path:
            y = layer(y)
        self.assertListEqual(list(y.shape), [data.shape[0],1, 1, 128], msg="down path y has unexpected output shape")
        for layer in self.network.up_path1:
            y = layer(y)
        self.assertListEqual(list(y.shape), [data.shape[0],9, 9, 64], msg="up path 1 y has unexpected output shape")
        y = self.network.concat([y,data])
        for layer in self.network.up_path2:
            y = layer(y)
        self.assertListEqual(list(y.shape), [data.shape[0],36, 36, 16], msg="up path 2 y has unexpected output shape")

        y = self.network.last(y)
        y = self.network.activation(y)
        self.assertListEqual(list(y.shape), [data.shape[0],72, 72, 3], msg="last y has unexpected output shape")


class TestBGMM(BaseTest):
    def setUp(self):
        self.network = CompressedSensingInceptionNet()

    def test_model(self):
        class MVNCholPrecisionTriL(tfd.TransformedDistribution):
            """MVN from loc and (Cholesky) precision matrix."""

            def __init__(self, loc, chol_precision_tril, name=None):
                super(MVNCholPrecisionTriL, self).__init__(
                    distribution=tfd.Independent(tfd.Normal(tf.zeros_like(loc),
                                                            scale=tf.ones_like(loc)),
                                                 reinterpreted_batch_ndims=1),
                    bijector=tfb.Chain([
                        tfb.Shift(shift=loc),
                        tfb.Invert(tfb.ScaleMatvecTriL(scale_tril=chol_precision_tril,
                                                       adjoint=True)),
                    ]),
                    name=name)

        def compute_sample_stats(d, seed=42, n=int(1e6)):
            x = d.sample(n, seed=seed)
            sample_mean = tf.reduce_mean(x, axis=0, keepdims=True)
            s = x - sample_mean
            sample_cov = tf.linalg.matmul(s, s, adjoint_a=True) / tf.cast(n, s.dtype)
            sample_scale = tf.linalg.cholesky(sample_cov)
            sample_mean = sample_mean[0]
            return [
                sample_mean,
                sample_cov,
                sample_scale,
            ]

        dtype = np.float32
        true_loc = np.array([1., -1.], dtype=dtype)
        true_chol_precision = np.array([[1., 0.],
                                        [2., 8.]],
                                       dtype=dtype)
        true_precision = np.matmul(true_chol_precision, true_chol_precision.T)
        true_cov = np.linalg.inv(true_precision)

        d = MVNCholPrecisionTriL(
            loc=true_loc,
            chol_precision_tril=true_chol_precision)

        [sample_mean, sample_cov, sample_scale] = [
            t.numpy() for t in compute_sample_stats(d)]


        print('true mean:', true_loc)
        print('sample mean:', sample_mean)
        print('true cov:\n', true_cov)
        print('sample cov:\n', sample_cov)

        dtype = np.float64
        dims = 2
        components = 3
        num_samples = 1000

        bgmm = tfd.JointDistributionNamed(dict(
            mix_probs=tfd.Dirichlet(
                concentration=np.ones(components, dtype) / 10.),
            loc=tfd.Independent(
                tfd.Normal(
                    loc=np.stack([
                        -np.ones(dims, dtype),
                        np.zeros(dims, dtype),
                        np.ones(dims, dtype),
                    ]),
                    scale=tf.ones([components, dims], dtype)),
                reinterpreted_batch_ndims=2),
            precision=tfd.Independent(
                tfd.WishartTriL(
                    df=5,
                    scale_tril=np.stack([np.eye(dims, dtype=dtype)] * components),
                    input_output_cholesky=True),
                reinterpreted_batch_ndims=1),
            s=lambda mix_probs, loc, precision: tfd.Sample(tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mix_probs),
                components_distribution=MVNCholPrecisionTriL(
                    loc=loc,
                    chol_precision_tril=precision)),
                sample_shape=num_samples)
        ))

        def joint_log_prob(observations, mix_probs, loc, chol_precision):
          """BGMM with priors: loc=Normal, precision=Inverse-Wishart, mix=Dirichlet.

          Args:
            observations: `[n, d]`-shaped `Tensor` representing Bayesian Gaussian
              Mixture model draws. Each sample is a length-`d` vector.
            mix_probs: `[K]`-shaped `Tensor` representing random draw from
              `Dirichlet` prior.
            loc: `[K, d]`-shaped `Tensor` representing the location parameter of the
              `K` components.
            chol_precision: `[K, d, d]`-shaped `Tensor` representing `K` lower
              triangular `cholesky(Precision)` matrices, each being sampled from
              a Wishart distribution.

          Returns:
            log_prob: `Tensor` representing joint log-density over all inputs.
          """
          return bgmm.log_prob(
              mix_probs=mix_probs, loc=loc, precision=chol_precision, s=observations)

        true_loc = np.array([[-2., -2],
                             [0, 0],
                             [2, 2]], dtype)
        random = np.random.RandomState(seed=43)

        true_hidden_component = random.randint(0, components, num_samples)
        observations = (true_loc[true_hidden_component] +
                        random.randn(num_samples, dims).astype(dtype))
        unnormalized_posterior_log_prob = functools.partial(joint_log_prob, observations)
        initial_state = [
            tf.fill([components],
                    value=np.array(1. / components, dtype),
                    name='mix_probs'),
            tf.constant(np.array([[-2., -2],
                                  [0, 0],
                                  [2, 2]], dtype),
                        name='loc'),
            tf.linalg.eye(dims, batch_shape=[components], dtype=dtype, name='chol_precision'),
        ]
        unconstraining_bijectors = [
            tfb.SoftmaxCentered(),
            tfb.Identity(),
            tfb.Chain([
                tfb.TransformDiagonal(tfb.Softplus()),
                tfb.FillTriangular(),
            ])]
        @tf.function(autograph=False)
        def sample():
          return tfp.mcmc.sample_chain(
            num_results=2000,
            num_burnin_steps=500,
            current_state=initial_state,
            kernel=tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn=unnormalized_posterior_log_prob,
                         step_size=0.065,
                         num_leapfrog_steps=5),
                    bijector=unconstraining_bijectors),
                 num_adaptation_steps=400),
            trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

        [mix_probs, loc, chol_precision], is_accepted = sample()
        acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32)).numpy()
        mean_mix_probs = tf.reduce_mean(mix_probs, axis=0).numpy()
        mean_loc = tf.reduce_mean(loc, axis=0).numpy()
        mean_chol_precision = tf.reduce_mean(chol_precision, axis=0).numpy()
        precision = tf.linalg.matmul(chol_precision, chol_precision, transpose_b=True)
        print('acceptance_rate:', acceptance_rate)
        print('avg mix probs:', mean_mix_probs)
        print('avg loc:\n', mean_loc)
        print('avg chol(precision):\n', mean_chol_precision)
        loc_ = loc.numpy()
        ax = sns.kdeplot(loc_[:,0,0], loc_[:,0,1], shade=True, shade_lowest=False)
        ax = sns.kdeplot(loc_[:,1,0], loc_[:,1,1], shade=True, shade_lowest=False)
        ax = sns.kdeplot(loc_[:,2,0], loc_[:,2,1], shade=True, shade_lowest=False)
        plt.title('KDE of loc draws');
        plt.show()




class TestDriftCorrectNet(tf.test.TestCase):
    def setUp(self):
        #todo: create test dataset with random on and off time and simulate a changing drift
        pass

    def test_layer_shape(self):
        pass

    def test_output(self):
        pass


def deprecated():
    @tf.function
    def vectorized_loss(data):
        x = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])  # [tf.newaxis,tf.newaxis,tf.newaxis,:]
        X, Y = tf.meshgrid(x, x)
        predict, truth = data
        loss = tf.constant(0, dtype=tf.float32)
        count = tf.constant(0, dtype=tf.float32)

        for i in range(3):
            loss += tf.cond(truth[i, 0] > 0, lambda: -tf.math.log(
                tf.reduce_sum(predict[:, :, 2] /
                              (tf.reduce_sum(predict[:, :, 2])
                               * tf.math.sqrt(tf.math.sqrt((predict[:, :, 3])) *
                                              tf.pow(2 * 3.14, 4) *
                                              tf.abs(tf.math.sqrt(predict[:, :, 4])))
                               )
                              * tf.exp(-(1 / 2 * tf.square(
                    predict[:, :, 0] + Y - truth[i, 0])  # todo: test that this gives expected values
                                         / predict[:, :, 3]
                                         + tf.square(predict[:, :, 1] + X - truth[i, 1])
                                         / predict[:, :, 4]  # todo: activation >= 0
                                         )))),
                            lambda: tf.constant(0, dtype=tf.float32))
        # count += tf.cond(truth_l[i][0] > 0, lambda: tf.constant(1,dtype=tf.float32), lambda:tf.constant(0,dtype=tf.float32))
        sigma_c = tf.reduce_mean(predict[:, :, 2] * (1 - predict[:, :, 2]))
        # loss += tf.square(count-tf.reduce_sum(predict[:, :, 2]))-tf.math.log(tf.sqrt(2*3.14*sigma_c))
        # /sigma_c-tf.math.log(tf.sqrt(2*3.14*sigma_c))
        return loss
    # L = tf.vectorized_map(vectorized_loss, [predict, truth])
    # L2 = tf.reduce_sum(tf.map_fn(vectorized_loss, (predict, truth), fn_output_signature=tf.float32))