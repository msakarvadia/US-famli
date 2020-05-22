from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json
import os
import glob
from . import base_nn
import sys

class NN(base_nn.BaseNN):

    def set_data_description(self, json_filename=None, data_description=None):
        super(NN, self).set_data_description(json_filename=json_filename, data_description=data_description)

        if("data_keys" in self.data_description 
            and len(self.data_description["data_keys"]) > 0
            and "max" in self.data_description[self.data_description["data_keys"][1]]):

            key_name_class = self.data_description["data_keys"][1]
            self.num_classes = int(self.data_description[self.data_description["data_keys"][1]]["max"])
            print("Setting the number of classes to", self.num_classes)
        else:
            print("Setting the number of classes to 2", file=sys.stderr)
            self.num_classes = 2

    def up_conv_block(self, x0, in_filters, out_filters, cross_block, block='a', is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        x = self.convolution2d(x0, name= block + "_conv1_op", filter_shape=[1,1,in_filters,out_filters[0]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)

        out_shape=tf.shape(cross_block)
        x = self.up_convolution2d(x, name=block + "_up_conv1_op", filter_shape=[3,3,out_filters[1],out_filters[0]], output_shape=out_shape, strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)

        x = tf.concat([cross_block, x], -1)
        x = self.convolution2d(x, name=block + "_conv2_op", filter_shape=[1,1,x.get_shape().as_list()[-1],out_filters[2]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)

        shortcut = self.up_convolution2d(x0, name=block + "_up_conv2_op", filter_shape=[3,3,out_filters[2],in_filters], output_shape=out_shape, strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        shortcut = tf.layers.batch_normalization(shortcut, training=is_training)

        x = tf.math.add(x, shortcut)
        x = tf.nn.leaky_relu(x)

        return x

    def inference(self, data_tuple=None, images=None, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        if(data_tuple):
            images = data_tuple[0]

    #   input: tensor of images
    #   output: tensor of computed logits
        self.print_tensor_shape(images, "images")

        shape = tf.shape(images)
        batch_size = shape[0]
            

        with tf.variable_scope("ued_resnet"):

            x0 = tf.layers.batch_normalization(images, training=is_training)

            with tf.variable_scope("block0"):
                x = self.convolution2d(x0, name="conv0_0_op", filter_shape=[3,3,4,32], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                block0_0_shape = tf.shape(x)
                x = self.convolution2d(x, name="conv1_0_op", filter_shape=[3,3,32,64], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                block0 = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block1"):
                x = self.conv_block(x, 64, [64,64,128], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 128, [64,64,128], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 128, [64,64,128], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                block1 = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block2"):
                x = self.conv_block(x, 128, [128,128,256], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 256, [128,128,256], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 256, [128,128,256], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 256, [128,128,256], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                block2 = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block3"):
                x = self.conv_block(x, 256, [256,256,512], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 512, [256,256,512], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 512, [256,256,512], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 512, [256,256,512], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 512, [256,256,512], block='e', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                block3 = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block4"):
                x = self.conv_block(x, 512, [512,512,1024], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='e', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='f', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("up_block4"):
                x = self.identity_block(x, 1024, [512,512,1024], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 1024, [512,512,1024], block='e', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.up_conv_block(x, 1024, [512,512,512], block='f', cross_block=block3, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("up_block3"):
                x = self.identity_block(x, 512, [256,256,512], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 512, [256,256,512], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 512, [256,256,512], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 512, [256,256,512], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.up_conv_block(x, 512, [256,256,256], block='e', cross_block=block2, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("up_block2"):
                x = self.identity_block(x, 256, [128,128,256], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 256, [128,128,256], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 256, [128,128,256], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.up_conv_block(x, 256, [128,128,128], block='d', cross_block=block1, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("up_block1"):
                x = self.identity_block(x, 128, [64,64,128], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.identity_block(x, 128, [64,64,128], block='b', activation=tf.nn.leaky_relu,is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.up_conv_block(x, 128, [64,64,64], block='c', cross_block=block0, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("up_block_final"):
                x = self.up_convolution2d(x, name="up_conv1_op", filter_shape=[3,3,32,64], output_shape=block0_0_shape, strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = self.up_convolution2d(x, name="up_conv2_op", filter_shape=[3,3,self.num_classes,32], output_shape=[shape[0], shape[1], shape[2], self.num_classes], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.nn.sigmoid(x)
            return x

    def metrics(self, logits, data_tuple, name='collection_metrics'):

         with tf.variable_scope(name):

            logits = tf.math.argmax(logits, axis=-1)
            labels = data_tuple[1]
            labels = tf.reshape(labels, tf.shape(logits))

            metrics_obj = {}

            metrics_obj["ACCURACY"] = tf.metrics.accuracy(predictions=logits, labels=labels, name='accuracy')
            # metrics_obj["AUC"] = tf.metrics.auc(predictions=logits, labels=labels, name='auc')
            # metrics_obj["FN"] = tf.metrics.false_negatives(predictions=logits, labels=labels, name='false_negatives')
            # metrics_obj["FP"] = tf.metrics.false_positives(predictions=logits, labels=labels, name='false_positives')
            # metrics_obj["TN"] = tf.metrics.true_negatives(predictions=logits, labels=labels, name='true_negatives')
            # metrics_obj["TP"] = tf.metrics.true_positives(predictions=logits, labels=labels, name='true_positives')
            
            for key in metrics_obj:
                tf.summary.scalar(key, metrics_obj[key][0])

            return metrics_obj

    def training(self, loss, learning_rate, decay_steps, decay_rate, staircase):
        
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # create learning_decay
        lr = tf.train.exponential_decay( learning_rate,
                                         global_step,
                                         decay_steps,
                                         decay_rate, staircase=staircase )

        tf.summary.scalar('2learning_rate', lr )

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(lr)

        vars_train = tf.trainable_variables()

        vars_ued = [var for var in vars_train if 'ued_resnet' in var.name]
        # Use the optimizer to apply the gradients that minimize the loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_ued)

        return train_op

    # def discriminator(self, images=None, data_tuple=None, num_labels=2, keep_prob=1, is_training=True, reuse=False, ps_device="/cpu:0", w_device="/gpu:0"):
    # #   input: tensor of images
    # #   output: tensor of computed logits
    #     if(data_tuple):
    #         images = data_tuple[0]
    #     self.print_tensor_shape(images, "images")

    #     shape = tf.shape(images)
    #     batch_size = shape[0]
        
    #     with tf.variable_scope("discriminator") as scope:
    #         if(reuse):
    #             scope.reuse_variables()

    #         x = tf.layers.batch_normalization(images, training=is_training)

    #         with tf.variable_scope("block0"):
    #             x = self.convolution2d(x, name="conv1", filter_shape=[5,5,self.num_channels,128], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
    #             x = tf.layers.batch_normalization(x, training=is_training)
    #             x = tf.nn.leaky_relu(x)
    #             x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

    #     return x

    def loss(self, logits, data_tuple, images=None, class_weights=None):
        
        labels = data_tuple[1]

        self.print_tensor_shape( logits, 'logits shape')
        self.print_tensor_shape( labels, 'labels shape')

        shape = tf.shape(logits)
        batch_size = shape[0]

        logits_flat = tf.reshape(logits, [batch_size, -1])

        labels = tf.one_hot(tf.cast(labels, tf.int32), 3)
        labels_flat = tf.reshape(tf.cast(labels, tf.float32), [batch_size, -1])

        intersection = 2.0 * tf.reduce_sum(logits_flat * labels_flat, axis=1) + 1e-7
        denominator = tf.reduce_sum(logits_flat, axis=1) + tf.reduce_sum(labels_flat, axis=1) + 1e-7

        return 1.0 - tf.reduce_mean(intersection / denominator)

    def predict(self, logits):
        return tf.math.argmax(logits, axis=-1)

    def prediction_type(self):
        return "segmentation"

    def restore_variables(self, restore_all=False):
        vars_train = tf.trainable_variables()
        if(restore_all):
            return vars_train
        return [var for var in vars_train if 'discriminator' in var.name]