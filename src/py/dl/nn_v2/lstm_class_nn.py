from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json
import os
import glob
import sys

class NN(tf.keras.Model):

    def __init__(self, tf_inputs, args):
        super(NN, self).__init__()
        
        learning_rate = args.learning_rate
        decay_steps = args.decay_steps
        decay_rate = args.decay_rate
        staircase = args.staircase
        drop_prob = args.drop_prob
        sample_weight = args.sample_weight

        data_description = tf_inputs.get_data_description()
        self.num_channels = data_description[data_description["data_keys"][1]]["shape"][-1]

        self.num_classes = 2
        if(data_description[data_description["data_keys"][1]]["num_class"]):
            self.num_classes = data_description[data_description["data_keys"][1]]["num_class"]
            print("Number of classes in data description", self.num_classes)

        self.drop_prob = drop_prob

        self.lstm_class = self.make_lstm_network()
        self.lstm_class.summary()

        # self.classification_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.sample_weight = None
        if(sample_weight is not None):
            self.sample_weight = tf.constant(np.reshape(np.multiply(np.ones([args.batch_size, self.num_classes]), sample_weight), [args.batch_size, self.num_classes, 1]), dtype=tf.float32)
            print("Weights:", self.sample_weight.numpy())
            
        self.metrics_acc = tf.keras.metrics.Accuracy()

        lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def make_lstm_network(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.BatchNormalization(input_shape=(64, 256, 256, 4)))
        model.add(tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), strides=(4, 4), activation='tanh', use_bias=False, padding='same'))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(), use_bias=False, padding='same'))
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(), use_bias=False, padding='same'))
        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(), use_bias=False, padding='same'))
        model.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(), use_bias=False, padding='same'))
        model.add(tf.keras.layers.Reshape(target_shape=(4*4*1024,)))
        model.add(tf.keras.layers.Dense(self.num_classes))

        return model


    @tf.function
    def train_step(self, train_tuple):
        
        images = train_tuple[0]
        labels = train_tuple[1]

        with tf.GradientTape() as tape:
            
            x_c = self.lstm_class(images)
            labels = tf.one_hot(labels, self.num_classes, axis=1)
            loss = self.classification_loss(labels, x_c, sample_weight=self.sample_weight)

            var_list = self.trainable_variables

            gradients = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(gradients, var_list))

            return loss, x_c

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(
            lstm_class=self.lstm_class,
            optimizer=self.optimizer)

    def summary(self, images, tr_step, step):
        labels = tf.reshape(images[1], [-1])

        loss = tr_step[0]
        prediction = tf.argmax(tf.nn.softmax(tr_step[1]), axis=1)

        self.metrics_acc.update_state(labels, prediction)
        acc_result = self.metrics_acc.result()

        print("step", step, "loss", loss.numpy(), "acc", acc_result.numpy(), labels.numpy(), prediction.numpy())
        tf.summary.image('real', images[0][:,32,:,:,:]/255, step=step)
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('accuracy', acc_result, step=step)

    def save_model(self, save_model):
        model = tf.keras.Sequential(self.lstm_class.layers + [layers.Lambda(lambda x: tf.sigmoid(x))])
        model.summary()
        model.save(save_model)