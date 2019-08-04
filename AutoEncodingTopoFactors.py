"""
Autoencoding Topographic Factors
"""

from __future__ import print_function

import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tfa_utils import plot, coordinate_tensor
from datetools import addDateTime

import pdb

class AutoEncodingTopoFactor(object):

    def __init__(self, dim_data, k, options, RLT_DIR, batch_size=5):

        # save shape of data
        self.dims = len(dim_data)
        self.dim_data = dim_data
        # save number of sources
        self.k = k
        # declare placeholders
        self.y_ph = tf.placeholder(tf.float32, [None]+self.dim_data)
        self.c_samps = tf.placeholder(tf.float32, [None, self.k, self.dims])
        # self.wi_samps = tf.placeholder(tf.float32, [None, self.k, 1])
        self.wi_samps = tf.placeholder(tf.float32, [None, self.k, self.dims*self.dims])
        self.we_samps = tf.placeholder(tf.float32, [None, self.k])
        self.training = tf.placeholder(tf.bool)
        self.dropout_ph = tf.placeholder(tf.float32)

        # determine which parameters are variational (i.e. sampled, if not point estimates are learned in the decoder)
        self.c_variational = True
        self.wi_variational = True
        self.we_variational = True

        # define initialization routines
        self.kernel_init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        self.weight_init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        self.bias_init = tf.constant_initializer(0.0)

        # regularization
        self.dropout_prob = 0.0

        # possible coordinates in data dimensions
        self.coordinates = self.init_coordinates()

        # build encoder network
        self.encoder_conv_layers = [7,5]
        self.encoder_full_layers = [1, .75]
        self.z_c_mean, self.z_c_std, self.z_wi_mean, self.z_wi_std, self.z_we_mean, self.z_we_std = self.encoder()

        # build decoder network
        self.decoder_conv_layers = []
        self.y_hat = self.decoder()

        # build loss operation
        self.loss_op = self.loss_operation()

        # configure training operation
        self.batch_size = options.batchsize
        self.learning_rate = options.learning_rate
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.num_epochs = options.maxepochs
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss_op,
                                                        global_step=self.global_step,
                                                        learning_rate=self.learning_rate,
                                                        optimizer=self.optimizer,
                                                        summaries=['loss','gradients'])

        # configure the tensorboard writer
        self.merged = tf.summary.merge_all()

        # set plt save directory
        self.save_dir = RLT_DIR
            #save_dir


    def init_coordinates(self):
        """
        Initialize coordinate system on the lattice
        """
        # generate list of all possible coordinates
        coordinates = coordinate_tensor(self.dim_data)

        # create array fo coordinates and the midpoints of the coordinate system
        coordinates = np.array(coordinates, dtype=np.float32)
        midpoints = np.array([d/2 for d in self.dim_data], dtype=np.float32)

        # create zero centered coordinate system
        coordinates = coordinates - midpoints
        return coordinates


    def conv_layer_encoder(self, x, k_size, out_channels, name):
        with tf.variable_scope(name) as scope:
            # get input channels
            in_channels = x.get_shape()[-1].value

            # declare the kernel: [height, width, input channels, output channels]
            k_dims = [k_size] * self.dims + [in_channels, out_channels]
            k = tf.get_variable('k', k_dims, initializer=self.kernel_init)

            # run the filter over the inputs with a strice od 1 for all dimensions
            c = tf.nn.convolution(x, k, 'SAME')

            # apply dropout
            keep_prob = 1.0 - self.dropout_ph
            c = tf.nn.dropout(c, keep_prob)

            # add biases before activation
            b = tf.get_variable('b', [out_channels], initializer=self.bias_init)
            z = tf.nn.bias_add(c,b)

            # apply activation
            a = tf.nn.tanh(z, name=scope.name)

            # down sample and normalize output for next CNN layer
            k = [1] + [3] * self.dims + [1]
            s = [1] + [2] * self.dims + [1]
            if self.dims == 2:
                max_pool = tf.nn.max_pool(a, ksize=k, strides=s, padding='SAME')
            elif self.dims ==3:
                max_pool = tf.nn.max_pool3d(a, ksize=k, strides=s, padding='SAME')

        return max_pool

    def dense_layer_encoder(self, x, in_out_dim_ratio, scope_name):

        # get input dimensions
        in_dim = x.get_shape()[1].value

        # compute output dimensions using ratio
        out_dim = np.round(in_dim * in_out_dim_ratio)

        # within the scope name
        with tf.variable_scope(scope_name) as scope:
            # define wieght and bias variables
            w = tf.get_variable('w', [in_dim, out_dim], initializer=self.weight_init)
            b = tf.get_variable('b', [out_dim], initializer=self.bias_init)

            # apply affine transform
            z = tf.matmul(x,w) + b

            # apply ReLU activation
            a = tf.nn.tanh(z, scope.name)

            # apply dropout
            keep_prob = 1.0 - self.dropout_ph
            a = tf.nn.dropout(a, keep_prob)

        return a

    def affine_layer(self, x, out_dim, scope_name, pos_out=False):

        # get input dimensions
        in_dim = x.get_shape()[1].value

        # within the same scope
        with tf.variable_scope(scope_name) as scope:

            # define weight and bias variables
            w = tf.get_variable('w', [in_dim, out_dim], initializer=self.weight_init)
            b = tf.get_variable('b', [out_dim], initializer=self.bias_init)

            # apply affine transform
            z = tf.matmul(x,w) + b

            # positive output
            if pos_out:

                # convert to Re+ with continous differentiability
                with tf.variable_scope('PosReAct') as scope:
                    z_pos = tf.cast(tf.greater_equal(z, tf.constant(0.0, dtype=tf.float32)), dtype=tf.float32)
                    z_neg = tf.cast(tf.less(z, tf.constant(0.0, dtype=tf.float32)), dtype=tf.float32)
                    a = z_neg * tf.exp(z) + z_pos * (z + tf.constant(1.0, dtype=tf.float32))

            # don't care
            else:
                a = z

        return a

    def encoder(self):

        # add a channel dimension to the input
        x = tf.expand_dims(self.y_ph, axis=-1)

        # normalize images
        # x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)

        # loop over the number of convolution layers
        for i in range(len(self.encoder_conv_layers)):
            x = self.conv_layer_encoder(x, self.encoder_conv_layers[i], self.k, 'EncConv%d' %i)

        # flatten features to vector
        x = tf.contrib.layers.flatten(x)

        # loop over the shared fully connected layers
        for i in range(len(self.encoder_full_layers)):
            # run cnn layers
            x = self.dense_layer_encoder(x, self.encoder_full_layers[i], 'EncFull%d' %i)

        # use name scope for better visualization
        with tf.variable_scope('Centers') as scope:
            # initialize z for centers
            z_c_mean = []
            z_c_std  = []

            # learning function approximator for variational centers?
            if self.c_variational:
                # loop oer the sources
                for i in range(self.k):
                    # run affine transform to characterize a source center's mean and std
                    z_c_mean.append(self.affine_layer(x, self.dims, 'c%d_mean' %i, pos_out=False))
                    z_c_std.append(self.affine_layer(x, self.dims, 'c%d_std' %i, pos_out=True))

                # stack them into tensors
                z_c_mean = tf.stack(z_c_mean)
                z_c_std = tf.stack(z_c_std)

        # use name scope for better visualization
        with tf.variable_scope('Widths') as scope:

            # initialize z for widths
            z_wi_mean = []
            z_wi_std = []

            # learning function approximator for variational widths?
            if self.wi_variational:

                # loop over the sources
                for i in range(self.k):
                    # run affine transform to characterize a sources width's mean and std
                    z_wi_mean.append(self.affine_layer(x, self.dims*self.dims, 'wi%d_mean' % i, pos_out=False))
                    z_wi_std.append(self.affine_layer(x, self.dims*self.dims, 'wi%d_std' % i, pos_out=True))

                # stack them into tensors
                z_wi_mean = tf.stack(z_wi_mean)
                z_wi_std = tf.stack(z_wi_std)

        # use name scope for better visualization
        with tf.variable_scope('Weights') as scope:

            # initialize z for weights
            z_we_mean = None
            z_we_std = None

            # learning function approximator for variational weights?
            if self.we_variational:
                # run affine transform to characterize a source weights mean and std
                z_we_mean = self.affine_layer(x, self.k, 'we_mean', pos_out=False)
                z_we_std = self.affine_layer(x, self.k, 'we_std', pos_out=True)

        # return latent variables
        return z_c_mean, z_c_std, z_wi_mean, z_wi_std, z_we_mean, z_we_std

    def compute_RBF_basis(self, c, w):

        # compute distance from center to all coordinates in the data dimensions
        d = self.coordinates - tf.expand_dims(c, axis=1)

        # compute squared norm along last dimension
        norm = tf.reduce_sum(d*d, axis=2)

        # apply RBF kernel
        f = tf.exp(-norm / (2 * (w * w + tf.constant(1.0))))

        # reshape [batch size, data dims]
        f =  tf.reshape(f, [-1] + self.dim_data)

        return f

    def compute_MVN_basis(self, c, w):

        # compute distance from center to all coordinates in the data dimensions
        d = self.coordinates - tf.expand_dims(c, axis=1)

        # compute precision of the cholesky decomposition
        TheChol = tf.reshape(w, [tf.shape(w)[0], self.dims, self.dims])
        InvSigm = tf.matmul(TheChol, tf.transpose(TheChol, perm=[0,2,1]))

        # compute the quadratic form
        QF = tf.einsum('abc,acd,abd->ab', d, InvSigm, d)

        # compute the unnormalized density
        f = tf.exp(-QF/(2*(1+tf.constant(1.0))))

        # reshape [batch size, data dims]
        f = tf.reshape(f, [-1] + self.dim_data)

        return f

    def matern(self, c, w):

        # compute distance from center to all coordinates in data dimensions
        d = self.coordinates - tf.expand_dims(c, axis=1)

        # compute squared norm along last dimension
        norm = tf.reduce_sum(d*d, axis=2)

        # Apply Matern
        partition = 2**(1-w)/tf.exp(tf.lgamma(w))
        return

    def conv_layer_decoder(self, x, k_size, out_channels, name):

        # define scope name
        with tf.variable_scope(name) as scope:
            # get input channels
            in_channels = x.get_shape()[-1].value

            # declare the kernel: [height, width, input channels, output channels]
            k_dims = [k_size] * self.dims + [in_channels, out_channels]
            k = tf.get_variable('k', k_dims, initializer=self.kernel_init)

            # run the filter over the inputs with a stride of 1 for all dimensions
            c = tf.nn.convolution(x, k, 'SAME')

            # apply dropout
            keep_prob = 1.0 - self.dropout_ph
            c = tf.nn.dropout(c, keep_prob)

            # add biases before activation
            b = tf.get_variable('b', [out_channels], initializer=self.bias_init)
            z = tf.nn.bias_add(c, b)

            # apply activation
            a = tf.nn.tanh(z, name=scope.name)

        return a

    def decoder(self):

        with tf.variable_scope('BasisFunction') as scope:

            # if centers are not variational parameters, declare as a trainable variable
            if not self.c_variational:
                c_var = tf.get_variable('c', [1, self.k, self.dims], initializer=self.weight_init)

            # if widths are not variational parameters, declare as a trainable variable
            if not self.wi_variational:
                wi_var = tf.get_variable('wi', [1, self.k], initializer=self.weight_init)

            # loop over the latent space to generate source images
            basis_images = []
            for i in range(self.k):

                # if centers are variational parameters, shift and scale this source's center sample
                if self.c_variational:
                    c = self.z_c_mean[i] + self.c_samps[:,i,:] * self.z_c_std[i]

                # centers are point estimates, slice out the kth source center
                else:
                    c = c_var[:,i,:]

                # if widths are variational parameters, shift and scale this source's width sample
                if self.wi_variational:
                    wi = self.z_wi_mean[i] + self.wi_samps[:,i,:] * self.z_wi_std[i]


                # if centers are point estimates, slice out the kth source width
                else:
                    wi = wi_var[:,i]

                # ensure width is positive
                with tf.variable_scope('PosReAct') as scope:
                    w_pos = tf.cast(tf.greater_equal(wi, tf.constant(0.0, dtype=tf.float32)), dtype=tf.float32)
                    w_neg = tf.cast(tf.less(wi, tf.constant(0.0, dtype=tf.float32)), dtype=tf.float32)
                    wi = w_neg * tf.exp(wi) + w_pos * (wi + tf.constant(1.0, dtype=tf.float32))

                # compute source basis image
                basis_images.append(self.compute_MVN_basis(c, wi))

            # stack basis images into a tensor [batch, {data dims}, k]
            x = tf.stack(basis_images)
            transpose = [i + 1 for i in range(1 + self.dims)] + [0]
            x = tf.transpose(x, transpose)

        # apply non-linear cnn layers
        for i in range(len(self.decoder_conv_layers)):
            x = self.conv_layer_decoder(x, self.decoder_conv_layers[i], self.k, 'DecConv%d' %i)

        with tf.variable_scope('1x1Conv') as scope:

            # variational weights?
            if self.we_variational:

                # sample weights
                we = self.z_we_mean + self.we_samps * self.z_we_std


                # add dimensions for compatibility
                for i in range(self.dims):
                    we = tf.expand_dims(we, axis=1)

                # apply weights
                y_hat = tf.reduce_sum(tf.multiply(x, we), axis=-1)

            # point estimation in decoder trainable parameters
            else:

                # run final 1x1 kernel to blend sources
                k_dims = [1] * self.dims + [self.k, 1]
                k = tf.get_variable('k', k_dims, initializer=self.kernel_init)
                x = tf.nn.convolution(x, k, 'SAME')
                y_hat = x

                # remove last dimensions
                y_hat = tf.squeeze(y_hat, axis=(self.dims + 1))

        return y_hat


    def loss_operation(self):
        """
        Defines the loss via decomposing the ELBO into two terms using Monte Carlo approximation of the expectation
        """

        loss = tf.squared_difference(self.y_ph, self.y_hat)
        loss = tf.reduce_mean(loss)

        # at least one variational parameter
        if False:

            # assemble lists of means and std for variational parameters
            z_mean = []
            z_std  = []
            if self.c_variational:
                z_mean.append(self.z_c_mean)
                z_std.append(self.z_c_std)
            if self.wi_variational:
                z_mean.append(self.z_wi_mean)
                z_std.append(self.z_wi_std)
            if self.z_we_variational:
                z_mean.append(tf.expand_dims(tf.transpose(self.z_we_mean), axis=-1))
                z_std.append(tf.expand_dims(tf.transpose(selfz_we_std), adis=-1))

            # convert to tensors of shape [None, Latent Space Dims]
            z_mean = tf.contrib.layers.flatten(tf.transpose(tf.concat(z_mean, axis=-1), [1,0,2]))
            z_std = tf.contrib.layers.flatten(tf.transpose(tf.concat(z_std, axis=-1), [1,0,2]))

            # compute entropy per sample
            lat_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_std) - tf.log(tf.square(z_std)) -1, axis=1)

            # take average over samples
            entropy = tf.reduce_mean(lat_loss)

        # no variational parameters (latent loss is zero)
        else:
            entropy = 0

        # compute total loss
        total_loss = loss + entropy
        return total_loss

    def feed_dict_samples(self, y_batch, training):

        # initialize feed dictionary
        feed_dict = dict()
        batch_size = y_batch.shape[0]

        # take M random samples for center and width from the standard normal N(0,I)
        c_MxKxD = np.random.normal(size=[batch_size, self.k, self.dims])
        wi_MxKxDxD = np.random.normal(size=[batch_size, self.k, self.dims*self.dims])
        we_MxK = np.random.normal(size=[batch_size, self.k])

        # load feed dictionary with data
        feed_dict.update({self.y_ph: y_batch})
        feed_dict.update({self.c_samps: c_MxKxD})
        feed_dict.update({self.wi_samps: wi_MxKxDxD})
        feed_dict.update({self.we_samps: we_MxK})

        # load feed dictionary with training hyper-parameters
        feed_dict.update({self.training: training})
        feed_dict.update({self.dropout_ph: self.dropout_prob * float(training)})

        return feed_dict

    def get_batches(self, data_len, shuffle=True):

        # get indices
        indices = np.arange(data_len)

        # shufle if specified
        if shuffle:
            np.random.shuffle(indices)

        # determine batch list
        batches = []
        while len(indices) > 0:
            batches.append(indices[:self.batch_size])
            indices = indices[self.batch_size:]

        # ensure al samples going through
        total_batch_len = sum([len(batch) for batch in batches])
        assert total_batch_len == data_len, "Missing elements!"

        return batches


    def train(self, y, yTest=None):

        # declare figure handles for plots updated during training
        fig_image = plt.figure()
        fig_loss = plt.figure()

        #print("yTest: ", yTest)

        with tf.Session() as sess:

            # run initialization
            sess.run(tf.global_variables_initializer())

            # configure writer if save directory is specified
            if self.save_dir is not None:
                train_writer = tf.summary.FileWriter(self.save_dir, sess.graph)
            else:
                train_writer = None

            # loop over the number of epochs
            loss_hist = []
            for i in range(self.num_epochs):

                # start timer
                start = time.time()

                # get training batches
                batches = self.get_batches(y.shape[0])

                # configure writer to write five times for this epoch
                write_interval = np.round(len(batches)/1)

                # loop over batches
                min_batch_loss = np.inf
                for j in range(len(batches)):

                    # load a feed dictionary
                    feed_dict = self.feed_dict_samples(y[batches[j]], True)

                    # writing to tensorboard?
                    if np.mod(i, write_interval) == 0 and train_writer is not None:

                        # run training, loss, accuracy
                        _, loss, summary = sess.run([self.train_op, self.loss_op, self.merged], feed_dict=feed_dict)

                        # write the summary
                        train_writer.add_summary(summary, self.global_step.eval(sess))

                    # not writing to tensorboard
                    else:
                        # run just training and loss
                        _, loss = sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)

                    # test for NaN
                    if np.isnan(loss):
                        print("NaN. Fuck.")
                        return

                    # save best loss for tis epoch
                    if loss < min_batch_loss:
                        min_batch_loss = loss

                    # print update
                    per = 100 * (j + 1) / len(batches)
                    update_str = 'Epoch %d, Percent Compete = %f%%, Cost = %f' % (i, per, loss)
                    print('\r' + update_str, end='')

                stop = time.time()
                print('\nTime for Epoch = %f' % (stop-start))
                # grab random observation
                i_plot = np.random.choice(y.shape[0])
                y_plot = np.expand_dims(y[i_plot], axis=0)

                # generate test image
                feed_dict = self.feed_dict_samples(y_plot, False)

                # run the optimizer
                y_hat = sess.run([self.y_hat], feed_dict=feed_dict)[0]
                y_hat.dump(self.save_dir + "/y_hat_%d" %i)



                # concatenate results
                y_plot = np.concatenate((y_plot, y_hat), axis=0)

                # plot image
                plot(y_plot,
                     fig=fig_image,
                     super_title=('Model_Performance_Epoch_%d' % i),
                     titles=['Scan %d' % i_plot, 'Estimate'],
                     plot_dir=self.save_dir)

                # plot the loss
                loss_hist.append(min_batch_loss)
                fig_loss.clf()
                sp = fig_loss.add_subplot(1,1,1)
                sp.plot(loss_hist, figure=fig_loss)
                sp.set_title('Objective: Min at Epoch = %d' % np.argmin(loss_hist))
                sp.set_xlabel('Epoch')
                sp.set_ylabel('Minimum Batch Loss for Epoch')
                if self.save_dir is None:
                    plt.pause(0.05)
                else:
                    fig_loss.savefig(self.save_dir + '/Learning_Curve')

            # evaluate parameters on train data
            scan_centers, scan_widths, scan_weights = self.save_params(sess, y) # these are not covariance matrices but components of Lambda

            #self.train_loss, self.train_mse = self.train_loss(sess, y)
            train_loss, train_mse = self.train_loss(sess, y)

            if yTest is not None:
                test_centers, test_widths, test_weights = self.save_params(sess, yTest)

                test_error = self.test_loss(sess, yTest)
                self.plot_test(sess, yTest)

                sess.close()
                return scan_centers, scan_widths, scan_weights, test_error, test_centers, test_widths, test_weights, train_loss, train_mse

            else:
                return scan_centers, scan_widths, scan_weights, train_loss, train_mse


    def save_params(self, sess, y):

        # initialize scan parameter arrays
        scan_centers = np.zeros([y.shape[0], self.k, self.dims])
        scan_widths = np.zeros([y.shape[0], self.k, self.dims * self.dims])
        scan_weights = np.zeros([y.shape[0], self.k])

        # loop over the scans
        for i in range(y.shape[0]):
            # evaluate scan
            feed_dict = self.feed_dict_samples(np.expand_dims(y[i], axis=0), False)
            center_means, width_means, weight_means = sess.run([self.z_c_mean, self.z_wi_mean, self.z_we_mean],
                                                               feed_dict=feed_dict)

            # load results
            scan_centers[i] = center_means[:, 0, :]
            scan_widths[i] = width_means[:, 0, :]
            scan_weights[i] = weight_means[0, :]

        # return to standard positive lattice positions
        mid_points = np.array([d / 2 for d in self.dim_data], dtype=np.float32)
        scan_centers = scan_centers + mid_points

        return scan_centers, scan_widths, scan_weights

    def train_loss(self, sess, yTrain):

        # import pdb
        # pdb.set_trace()
        # returns the loss on the validation dataset
        error = []

        for i in range(yTrain.shape[0]):
            feed_dict = self.feed_dict_samples(np.expand_dims(yTrain[i], axis=0), False)
            # y_hat = sess.run([self.y_hat], feed_dict=feed_dict)[0]
            loss = sess.run([self.loss_op], feed_dict=feed_dict)
            error.append(loss[0])

        return np.asarray(error), np.mean(error)


    def test_loss(self, sess, yTest):

        # import pdb
        # pdb.set_trace()
        # returns the loss on the validation dataset
        error = []

        for i in range(yTest.shape[0]):
            #pdb.set_trace()
            feed_dict = self.feed_dict_samples(np.expand_dims(yTest[i], axis=0), False)
            # y_hat = sess.run([self.y_hat], feed_dict=feed_dict)[0]
            loss = sess.run([self.loss_op], feed_dict=feed_dict)
            error.append(loss[0])

        return np.asarray(error)
    
    

    def plot_test(self, sess, yTest):

        print("plotting test reconstructions...")

        fig_image = plt.figure()

        for i in range(yTest.shape[0]):
            feed_dict = self.feed_dict_samples(np.expand_dims(yTest[i], axis=0), False)
            y_hat = sess.run([self.y_hat], feed_dict=feed_dict)[0]  # not sure if this is correct...
            y_plot = np.expand_dims(yTest[i], axis=0)

            # concatenate results
            y_plot = np.concatenate((y_plot, y_hat), axis=0)

            # plot image
            plot(y_plot,
                 fig=fig_image,
                 super_title=('Reconstructed_Image_%d' % i),
                 titles=['Scan %d' % i, 'Estimate'],
                 plot_dir=self.save_dir)

        return






