import tensorflow as tf
import datetime
import os
import sys
import argparse
import IPython
import _pickle as pickle

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.max_iter = 5000
        self.summary_iter = 200
        self.learning_rate = 0.1
        self.saver = tf.train.Saver()
        self.summary_op = tf.summary.merge_all()

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        '''
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        '''
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.net.class_loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def optimize(self):

        self.train_losses = []
        self.test_losses = []

        '''
        Performs the training of the network.
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test accuracy through out the process
        '''

        for step in range(1, self.max_iter + 1):
            images, labels = self.data.get_train_batch()
            feed_dict = {self.net.images: images, self.net.labels: labels}
            print("got training batch",step)
            if step % self.summary_iter == 0:
                loss = self.sess.run(
                    self.net.accurracy,
                    feed_dict=feed_dict)
                self.train_losses.append(loss)
                images_t, labels_t = self.data.get_validation_batch()
                feed_dict_test = {self.net.images : images_t, self.net.labels: labels_t}
                test_loss = self.sess.run(
                    self.net.accurracy,
                    feed_dict=feed_dict_test)


                self.test_losses.append(test_loss)
                if test_loss > 0.6:
                    return
            else:
                self.sess.run([self.train],feed_dict=feed_dict)

    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            self.cfg_dict = self.cfg.__dict__
            for key in sorted(self.cfg_dict.keys()):
                if key[0].isupper():
                    self.cfg_str = '{}: {}\n'.format(key, self.cfg_dict[key])
                    f.write(self.cfg_str)
