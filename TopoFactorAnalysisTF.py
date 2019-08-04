from __future__ import print_function
#from cloud_run import cloud_run

# cloud run?
#if cloud_run():
#    import matplotlib as mpl
#    mpl.use('Agg')

import os
import shutil
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
#from data_generator import DataGenerator
from hotspot_init import hot_spot_init
from tfa_utils import log_precision_to_variance, rbf_basis_image, plot, coordinate_tensor
import time
import pickle
import pdb

class TFA:

    def __init__(self, y, k, theta_init, center_init=None, width_init=None, save_dir=None):

        # save image and its min/max
        self.y = y
        self.y_min = np.min(y)
        self.y_max = np.max(y)

        # describe the data hyper-cube
        self.n = y.shape[0]
        self.d = len(y.shape[1:])
        self.dim = y.shape[1:]
        self.v = np.prod(self.dim)

        # configure number of sources
        self.k = k

        # configure batch sizes
        self.mc_samples = 30

        # configure input data placeholder and load it with the feed dictionary
        self.y_placeholder = tf.placeholder(tf.float32, [self.n, self.v])
        self.feed_dict = dict()
        self.feed_dict.update({self.y_placeholder: np.reshape(y, [-1, self.v])})

        # configure monte carlo samples input placeholder
        self.w_samples = tf.placeholder(tf.float32, [None, self.n, self.k])
        self.m_samples = tf.placeholder(tf.float32, [None, self.k, self.d])
        self.l_samples = tf.placeholder(tf.float32, [None, self.k])

        # initialize theta (generative model)
        self.theta, theta_init = self.init_theta(theta_init, center_init)

        # initialize alpha (recognition model)
        self.alpha = self.init_alpha(y, theta_init, center_init, width_init)

        # initialize coordinate lists
        self.c = self.init_coordinates()

        # configure training
        self.learning_rate = 1e-6
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #tf.train.GradientDescentOptimizer(self.learning_rate)
        #tf.train.AdamOptimizer(self.learning_rate)#MomentumOptimizer(self.learning_rate, momentum=0.9)
        self.max_iterations = 100#500
        self.max_no_improvement = 6

        # set global step variable for optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # set plot save directory
        self.save_dir = save_dir

        # configure loss and training operations
        self.loss_op = self.loss_operation()
        self.train_op = self.train_operation()
        self.image_op = self.image_operation()

        # configure the tensor board writer
        self.merged = tf.summary.merge_all()

    def init_theta(self, theta_init, center_init=None):

        # initialize pi to empty dictionary
        theta = dict()

        # loop over the values in the theta initializer
        for key, value in theta_init.items():
            theta.update({key: tf.constant(value, dtype=tf.float32)})

        # were centers provided?
        if center_init is not None:

            # compute mean of distribution over source centers
            c = np.mean(center_init, axis=0)
            mu_var = np.var(center_init, axis=0)
            mu_var[mu_var <= 0] = 1

            # update hyper-parameter dictionary
            theta.update({'mu_var': tf.constant(mu_var, dtype=tf.float32)})
            theta.update({'center_mean': tf.constant(c, dtype=tf.float32)})
            theta.update({'center_log_pre': tf.constant(-np.log(10 * mu_var), dtype=tf.float32)})

            # update the initialization copy
            theta_init.update({'mu_var': mu_var})
            theta_init.update({'center_mean': c})
            theta_init.update({'center_log_pre': -np.log(10 * mu_var)})

        return theta, theta_init

    def init_alpha(self, y, theta_init, center_init, width_init):

        # save if center and width initializations were provided
        both_provided = (center_init is not None) and (width_init is not None)

        # initialize alpha to empty dictionary
        alpha = dict()

        # center initialization not provided
        if center_init is None:

            # sample center initialization from prior
            center_init = np.random.normal(theta_init['center_mean'],
                                           np.sqrt(log_precision_to_variance(theta_init['center_log_pre'])),
                                           [self.k, self.d])

        # center variational parameters
        alpha['center'] = {
            # initialize mean
            'mean': tf.Variable(center_init,
                                dtype=tf.float32,
                                name='center_mean'),

            # initialize log precision
            'log_pre': tf.Variable(np.log(100 / theta_init['mu_var']) * np.ones([self.k, self.d]),
                                   dtype=tf.float32,
                                   name='center_log_precision')
        }

        # width initialization not provided
        if width_init is None:

            # sample width initialization from prior
            width_init = np.random.normal(theta_init['width_mean'],
                                          np.sqrt(log_precision_to_variance(theta_init['width_log_pre'])),
                                          [self.k])

        # width variational parameters
        alpha['width'] = {
            # initialize mean
            'mean': tf.Variable(width_init,
                                dtype=tf.float32,
                                name='width_mean'),

            # initialize log precision
            'log_pre': tf.Variable(np.ones([self.k]),
                                   dtype=tf.float32,
                                   name='width_log_precision')
        }

        # generate basis image using center and weight initializations
        f = rbf_basis_image(center_init, width_init, self.dim)

        # both weight and width initializations provided
        if both_provided:

            # flatten y and f
            y = np.reshape(y, [y.shape[0], -1])
            f = np.reshape(f, [f.shape[0], -1])

            try:
                # solve for weight initialization
                #weight_init = y @ np.linalg.pinv(f)
                weight_init = np.dot(y, np.linalg.pinv(f))

            except:
                # sample weight initialization from prior
                weight_init = np.random.normal(theta_init['weight_mean'],
                                               np.sqrt(log_precision_to_variance(theta_init['weight_log_pre'])),
                                               [self.n, self.k])

        # both not provided
        else:

            # sample weight initialization from prior
            weight_init = np.random.normal(theta_init['weight_mean'],
                                           np.sqrt(log_precision_to_variance(theta_init['weight_log_pre'])),
                                           [self.n, self.k])

        # weight variational parameters
        alpha['weight'] = {
            # initialize mean
            'mean': tf.Variable(weight_init,
                                dtype=tf.float32,
                                name='weight_mean'),

            # initialize log precision
            'log_pre': tf.Variable(np.log(10) * np.ones([self.n, self.k]),
                                   dtype=tf.float32,
                                   name='weight_log_precision')
        }

        return alpha

    def init_coordinates(self):

        # generate list of all possible coordinates
        c = coordinate_tensor(self.dim)

        # create tensor
        c = np.array(c)

        return c

    @staticmethod
    def log_precision_to_variance(k):

        # convert to variance
        var = tf.reciprocal(tf.exp(k))

        return var

    def sample_q(self, w_samps, m_samps, l_samps):

        # sample weights
        std_matrix = tf.sqrt(self.log_precision_to_variance(self.alpha['weight']['log_pre']))
        w = self.alpha['weight']['mean'] + tf.multiply(std_matrix, w_samps)

        # sample centers
        std_matrix = tf.sqrt(self.log_precision_to_variance(self.alpha['center']['log_pre']))
        m = self.alpha['center']['mean'] + tf.multiply(std_matrix, m_samps)

        # sample widths
        std_matrix = tf.sqrt(self.log_precision_to_variance(self.alpha['width']['log_pre']))
        l = self.alpha['width']['mean'] + tf.multiply(std_matrix, l_samps)

        return w, m, l

    def log_q(self, w, m, l):

        # configure weights distribution
        mean = tf.reshape(self.alpha['weight']['mean'], [-1])
        std = tf.sqrt(self.log_precision_to_variance(tf.reshape(self.alpha['weight']['log_pre'], [-1])))
        w_dist = tf.contrib.distributions.MultivariateNormalDiag(mean, std)

        # flatten weight samples to [num_samples, N * K]
        w = tf.contrib.layers.flatten(w)

        # evaluate log probability for each sample
        log_q_w = w_dist.log_prob(w)

        # configure centers distribution
        mean = tf.reshape(self.alpha['center']['mean'], [-1])
        std = tf.sqrt(self.log_precision_to_variance(tf.reshape(self.alpha['center']['log_pre'], [-1])))
        m_dist = tf.contrib.distributions.MultivariateNormalDiag(mean, std)

        # flatten centers samples to [num_samples, K * D]
        m = tf.contrib.layers.flatten(m)

        # evaluate log probability for each sample
        log_q_m = m_dist.log_prob(m)

        # configure width distribution
        mean = self.alpha['width']['mean']
        std = tf.sqrt(self.log_precision_to_variance(self.alpha['width']['log_pre']))
        l_dist = tf.contrib.distributions.MultivariateNormalDiag(mean, std)

        # evaluate log probability for each sample
        log_q_l = l_dist.log_prob(l)

        # compute log q
        log_q = log_q_w + log_q_m + log_q_l

        return log_q

    def compute_y_hat(self, w, m, l):

        # expand dimensions on m to make it broadcast
        m = tf.expand_dims(m, axis=2)

        # computing all the distances
        d = self.c - m

        # compute squared norm along last dimension
        norm = tf.reduce_sum(d * d, axis=3)
        norm = tf.transpose(norm, [2, 0, 1])

        # apply RBF kernel
        f = tf.exp(-norm / self.log_precision_to_variance(l))

        # transpose to [num samples, num sources, num voxels]
        f = tf.transpose(f, [1, 2, 0])

        # compute y hat [num samples, num images, num voxels]
        y_hat = tf.matmul(w, f)

        return y_hat

    def log_p(self, y, w, m, l):

        # flatten y to a vector [num_images x num_voxels]
        y = tf.reshape(y, [-1])

        # compute estimate from weights, centers, and widths
        y_hat = self.compute_y_hat(w, m, l)

        # configure image distribution
        y_mean = tf.contrib.layers.flatten(y_hat)
        y_std = tf.fill([self.n * self.v], tf.sqrt(self.theta['y_var']))

        #y_dist = tf.contrib.distributions.MultivariateNormalDiag(scale_diag=y_std) ###
        y_dist = tf.contrib.distributions.MultivariateNormalFull(mu=tf.zeros(self.n*self.v, dtype=tf.float32), sigma=tf.diag(y_std))

        # evaluate log probability
        log_p_y = y_dist.log_prob(y - y_mean)

        # configure weights distribution
        w_std = tf.fill([self.n * self.k], tf.sqrt(self.log_precision_to_variance(self.theta['weight_log_pre'])))
        #w_dist = tf.contrib.distributions.MultivariateNormalDiag(scale_diag=w_std) ###
        w_dist = tf.contrib.distributions.MultivariateNormalFull(mu=tf.zeros(self.n*self.k, dtype=tf.float32), sigma=tf.diag(w_std))

        # evaluate log probability
        w = tf.contrib.layers.flatten(w)
        log_p_w = w_dist.log_prob(w - self.theta['weight_mean'])

        # configure centers distribution
        m_std = tf.sqrt(self.log_precision_to_variance(self.theta['center_log_pre']))
        #m_dist = tf.contrib.distributions.MultivariateNormalDiag(scale_diag=m_std)  ###
        m_dist = tf.contrib.distributions.MultivariateNormalFull(mu=tf.zeros(self.d, dtype=tf.float32), sigma=tf.diag(m_std))  ###

        # evaluate log probability
        log_p_m = tf.reduce_sum(m_dist.log_prob(m - self.theta['center_mean']), axis=1)

        # configure widths distribution
        l_std = tf.sqrt(self.log_precision_to_variance(self.theta['width_log_pre']))
        #pdb.set_trace()
        #l_dist = tf.contrib.distributions.Normal(loc=tf.constant(0, dtype=tf.float32), scale=l_std)
        l_dist = tf.contrib.distributions.Normal(mu=tf.constant(0, dtype=tf.float32), sigma=l_std)

        # evaluate log probability
        log_p_l = tf.reduce_sum(l_dist.log_prob(l - self.theta['width_mean']), axis=1)

        # compute log p
        log_p = log_p_y + log_p_w + log_p_m + log_p_l

        return log_p

    def loss_operation(self):

        # sample multiple samples from q
        w, m, l = self.sample_q(self.w_samples, self.m_samples, self.l_samples)

        # evaluate log q(W, M, L | alpha) for each sample
        log_q = self.log_q(w, m, l)

        # evaluate loq p(Y, W, M, L) for each sample
        log_p = self.log_p(self.y_placeholder, w, m, l)

        # compute loss as the Monte Carlo estimate from the samples
        # note this is the negative from the paper since tensor flow seeks to minimize
        loss = tf.reduce_mean(log_q - log_p)

        return loss

    def train_operation(self):

        # # get training operation
        # train_op = tf.contrib.layers.optimize_loss(loss=self.loss_op,
        #                                            global_step=self.global_step,
        #                                            learning_rate=self.learning_rate,
        #                                            optimizer=self.optimizer,
        #                                            summaries=['loss', 'gradients'])

        # set the training operation for this estimate
        train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step)

        return train_op

    def image_operation(self):

        # sample once from q
        w, m, l = self.sample_q(self.w_samples, self.m_samples, self.l_samples)

        # compute fake image from weights, centers, and widths
        y = self.compute_y_hat(w, m, l)

        # reshape it from [N x voxels] to [N x d1, ..., dn]
        y = tf.reshape(y, shape=self.y.shape)

        return y

    def return_params(self):

        # sample from q
        w, m, l = self.sample_q(self.w_samples, self.m_samples, self.l_samples)

        # compute reconstruction
        y = self.compute_y_hat(w, m, l)

        # reshape
        y = tf.reshape(y, shape=self.y.shape)

        return w, m, l, y

    def feed_dict_samples(self, num_samples=None):

        # if num samples is not provided, set to Monte Carlo
        if num_samples is None:
            num_samples = self.mc_samples

        # sample N(0,1) into feed dictionary
        self.feed_dict.update({self.w_samples: np.random.normal(size=[num_samples, self.n, self.k])})
        self.feed_dict.update({self.m_samples: np.random.normal(size=[num_samples, self.k, self.d])})
        self.feed_dict.update({self.l_samples: np.random.normal(size=[num_samples, self.k])})

    def train(self):

        # start session
        with tf.Session() as sess:

            # run initialization
            sess.run(tf.global_variables_initializer())

            # # configure writer
            # if self.save_dir is not None:
            #     train_writer = tf.summary.FileWriter(self.save_dir, sess.graph)

            # turn on interactive plotting if save directory not specified
            if self.save_dir is None:
                plt.ion()

            # plot original image
            plot(self.y[0:5], super_title='Original_Data', plot_dir=self.save_dir)

            # generate pre-training image
            self.feed_dict_samples(1)
            y_prior = sess.run([self.image_op], feed_dict=self.feed_dict)
            plot(y_prior[0][0:5], super_title='A_Priori', plot_dir=self.save_dir)

            # declare figure handles for plots updated during training
            fig_learning = plt.figure()
            fig_generate = plt.figure()

            # loop the optimizer
            variational_objective = []
            no_improvement_count = 0
            for i in range(self.max_iterations):

                print("\niter:", i)

                start = time.time()
                # run the optimizer
                self.feed_dict_samples(self.mc_samples)
                _, loss = sess.run([self.train_op, self.loss_op], feed_dict=self.feed_dict)
                print("LOSS: ", loss)
                #import pdb
                #pdb.set_trace()
                #self.feed_dict_samples(1)
                #pars = sess.run([self.return_params()], feed_dict=self.feed_dict)
                
                

                # test for NaN and exit if so
                if np.isnan(loss):
                    print("NaN, bitches")
                    return

                # # run the optimizer
                # _, loss, summary = sess.run([self.train_op, self.loss_op, self.merged], feed_dict=feed_dict)
                #
                # # write the summary
                # if self.save_dir is not None:
                #     train_writer.add_summary(summary, self.global_step.eval(sess))

                # print/plot the loss
                variational_objective.append(-loss)
                fig_learning.clf()
                sp = fig_learning.add_subplot(1, 1, 1)
                sp.plot(variational_objective, figure=fig_learning)
                sp.set_title('Objective: Max at Epoch = %d' % np.argmax(variational_objective))
                sp.set_xlabel('Iteration')
                sp.set_ylabel('E.L.B.O.')
                if self.save_dir is None:
                    plt.pause(0.05)
                else:
                    fig_learning.savefig(self.save_dir + 'Learning_Curve')

                # generate post-training image
                self.feed_dict_samples(1)
                y_post = sess.run([self.image_op], feed_dict=self.feed_dict)
                #import pdb
                #pdb.set_trace()
                plot(y_post[0][0:5], fig=fig_generate, super_title=('A_Posteriori_Iteration_%d' % i), plot_dir=self.save_dir)

                # # check for no improvement
                # if len(cost) >= 2:
                #     if cost[-1] < cost[-2]:
                #         no_improvement_count += 1
                #     else:
                #         no_improvement_count = 0

                # early stop?
                if no_improvement_count >= self.max_no_improvement:
                    print('Early stop!')
                    break

                # print update
                percent_done = 100 * (i + 1.0) / self.max_iterations
                update_str = 'Percent Complete = %f%%, ELBO = %f' % (percent_done, -loss)
                stop = time.time()
                print('\nTime for iteration = %f' % (stop-start))
                print('\r' + update_str, end='')
                #import pdb
                #pdb.set_trace()

            # close session
            #import pdb
            #pdb.set_trace()
            #w_params, m_params, l_params, y_fit = self.return_params(self, sess)
            self.feed_dict_samples(1)
            pars = sess.run([self.return_params()], feed_dict=self.feed_dict)
            tfa_params = {
                'weights': pars[0][0],
                'centers': pars[0][1],
                'widths': pars[0][2],
                'yhat': pars[0][3],
                'ytrain': self.y,
                'k': self.k
            }
            print("Saving Parameters")
            pickle.dump(tfa_params, open(self.save_dir + 'tfa_model_params.p', 'wb'))

            sess.close()

        # print completion
        print('\nDone!')

        if self.save_dir is None:
            plt.ioff()
            plt.show()
