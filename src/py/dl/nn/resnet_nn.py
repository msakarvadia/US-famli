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

        if "data_keys" in self.data_description:
            data_keys = self.data_description["data_keys"]
            if data_keys[0] in self.data_description and "shape" in self.data_description[data_keys[0]]:
                self.num_channels = self.data_description[data_keys[0]]["shape"][-1]

        if("enumerate" in self.data_description and "num_class" in self.data_description[self.data_description["enumerate"]]):
            self.enumerate = self.data_description["enumerate"]
            self.num_classes = self.data_description[self.enumerate]["num_class"]
        else:
            print("You may want to call the tfRecords.py script with the flag --enumerate", key_name_class, file=sys.stderr)
            print("This option will count the number of classes in your data and enumerate accordingly", file=sys.stderr)
            print("Setting the number of classes to 2", file=sys.stderr)
            self.num_classes = 2

    def conv_block(self, x0, in_filters, out_filters, block='a', is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):
        
        x = self.convolution2d(x0, name= block + "_conv1_op", filter_shape=[1,1,in_filters,out_filters[0]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)

        x = self.convolution2d(x, name=block + "_conv2_op", filter_shape=[3,3,out_filters[0],out_filters[1]], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)

        x = self.convolution2d(x, name=block + "_conv3_op", filter_shape=[1,1,out_filters[1],out_filters[2]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)     

        shortcut = self.convolution2d(x0, name=block + "_conv4_op", filter_shape=[1,1,in_filters,out_filters[2]], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)    
        shortcut = tf.layers.batch_normalization(shortcut, training=is_training)

        x = tf.math.add(x, shortcut)
        x = tf.nn.relu(x)

        return x

    def identity_block(self, x0, in_filters, out_filters, block='a', is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        x = self.convolution2d(x0, name=block + "_conv1_op", filter_shape=[1,1,in_filters,out_filters[0]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)   

        x = self.convolution2d(x, name=block + "_conv2_op", filter_shape=[3,3,out_filters[0],out_filters[1]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)  

        x = self.convolution2d(x, name=block + "_conv3_op", filter_shape=[1,1,out_filters[1],out_filters[2]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x) 
        
        x = tf.math.add(x, x0)
        x = tf.nn.relu(x)

        return x

    def inference(self, data_tuple=None, images=None, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        if(data_tuple):
            images = data_tuple[0]

    #   input: tensor of images
    #   output: tensor of computed logits
        self.print_tensor_shape(images, "images")

        shape = tf.shape(images)
        batch_size = shape[0]

        images = tf.layers.batch_normalization(images, training=is_training)

        with tf.variable_scope("block0"):
            x = self.convolution2d(images, name="conv0_0_op", filter_shape=[7,7,self.num_channels,64], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            x = self.max_pool(x, name="pool0_0_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        with tf.variable_scope("block1"):
            x = self.conv_block(x, 64, [64,64,128], block='a', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 128, [64,64,128], block='b', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 128, [64,64,128], block='c', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = tf.nn.dropout(x, keep_prob)

        with tf.variable_scope("block2"):
            x = self.conv_block(x, 128, [128,128,256], block='a', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 256, [128,128,256], block='b', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 256, [128,128,256], block='c', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 256, [128,128,256], block='d', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = tf.nn.dropout(x, keep_prob)

        with tf.variable_scope("block3"):
            x = self.conv_block(x, 256, [256,256,512], block='a', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 512, [256,256,512], block='b', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 512, [256,256,512], block='c', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 512, [256,256,512], block='d', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 512, [256,256,512], block='e', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = tf.nn.dropout(x, keep_prob)

        with tf.variable_scope("block4"):
            x = self.conv_block(x, 512, [512,512,1024], block='a', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 1024, [512,512,1024], block='b', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 1024, [512,512,1024], block='c', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 1024, [512,512,1024], block='d', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 1024, [512,512,1024], block='e', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = self.identity_block(x, 1024, [512,512,1024], block='f', is_training=is_training, ps_device=ps_device, w_device=w_device)
            x = tf.nn.dropout(x, keep_prob)
        
        with tf.variable_scope("fc"):
            kernel_size = x.get_shape().as_list()
            kernel_size[0] = 1
            kernel_size[-1] = 1
            x = self.avg_pool(x, name="avg_pool_op", kernel=kernel_size, strides=kernel_size, ps_device=ps_device, w_device=w_device)
            x = tf.reshape(x, (batch_size, 1024))

            x = self.matmul(x, self.num_classes, name='final_op', activation=None, ps_device=ps_device, w_device=w_device)

        return x

    def metrics(self, logits, data_tuple, name='collection_metrics'):

        labels = data_tuple[1]
        with tf.variable_scope(name):
            weight_map = None

            logits_auc = tf.nn.softmax(logits)
            logits = tf.argmax(logits_auc, axis=1)

            labels_auc = tf.one_hot(labels, self.num_classes, axis=1)

            self.print_tensor_shape( logits, 'logits metrics shape')
            self.print_tensor_shape( labels, 'labels metrics shape')

            labels_auc = tf.reshape(labels_auc, tf.shape(logits_auc))
            self.print_tensor_shape( logits_auc, 'logits_auc metrics shape')
            self.print_tensor_shape( labels_auc, 'labels_auc metrics shape')
            
            metrics_obj = {}
            
            metrics_obj["ACCURACY"] = tf.metrics.accuracy(predictions=logits, labels=labels, weights=weight_map, name='accuracy')
            metrics_obj["AUC"] = tf.metrics.auc(predictions=logits_auc, labels=labels_auc, weights=weight_map, name='auc')
            metrics_obj["FN"] = tf.metrics.false_negatives(predictions=logits, labels=labels, weights=weight_map, name='false_negatives')
            metrics_obj["FP"] = tf.metrics.false_positives(predictions=logits, labels=labels, weights=weight_map, name='false_positives')
            metrics_obj["TN"] = tf.metrics.true_negatives(predictions=logits, labels=labels, weights=weight_map, name='true_negatives')
            metrics_obj["TP"] = tf.metrics.true_positives(predictions=logits, labels=labels, weights=weight_map, name='true_positives')
            metrics_obj["RECALL"] = tf.metrics.recall(predictions=logits, labels=labels, name="recall")
            # metrics_obj["PRECISION"] = [metrics_obj["TN"][0]/(metrics_obj["TN"][0] + metrics_obj["FP"][0])]
            # logits_perm = tf.transpose(logits_auc, perm=[1, 0]) 
            # labels_perm = tf.transpose(labels_auc, perm=[1, 0])

            # acc_per_class = tf.map_fn(lambda num_class: self.metrics_per_class(logits=logits_perm[num_class], labels=labels_perm[num_class], num_class=num_class), tf.range(self.num_classes), dtype=tf.float32)
            # metrics_obj["MEAN_PER_CLASS_ACCURACY"] = acc_per_class

            for key in metrics_obj:
                tf.summary.scalar(key, metrics_obj[key][1])

            return metrics_obj

    def metrics_per_class(self, logits, labels, num_class):
        self.print_tensor_shape( logits, 'logits per class metrics shape')
        self.print_tensor_shape( labels, 'labels per class metrics shape')

        shape = tf.shape(logits)
        batch_size = shape[0]

        labels_flat = tf.reshape(labels, [batch_size, -1])
        logits_flat = tf.reshape(logits, [batch_size, -1])

        return tf.metrics.mean_iou(predictions=logits_flat, labels=labels_flat)

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

        # Use the optimizer to apply the gradients that minimize the loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def loss(self, logits, data_tuple, class_weights=None):
        
        labels = data_tuple[1]

        self.print_tensor_shape( logits, 'logits shape')
        self.print_tensor_shape( labels, 'labels shape')

        return tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    def predict(self, logits):
        soft_logits = tf.nn.softmax(logits)
        return soft_logits

    def prediction_type(self):
        return "class"