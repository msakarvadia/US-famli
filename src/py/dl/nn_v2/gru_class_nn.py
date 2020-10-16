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

        data_description = tf_inputs.get_data_description()
        self.num_channels = data_description[data_description["data_keys"][0]]["shape"][-1]

        self.num_classes = 2
        self.class_weights_index = -1
        self.enumerate_index = 1

        if "enumerate" in data_description:
            self.enumerate_index = data_description["data_keys"].index(data_description["enumerate"])

            if(data_description[data_description["data_keys"][self.enumerate_index]]["num_class"]):
                self.num_classes = data_description[data_description["data_keys"][self.enumerate_index]]["num_class"]
                print("Number of classes in data description", self.num_classes)
                if "class_weights" in data_description["data_keys"]:
                    self.class_weights_index = data_description["data_keys"].index("class_weights")
                    print("Using weights index", self.class_weights_index)

        self.drop_prob = drop_prob

        self.lstm_class = self.make_lstm_network()
        self.lstm_class.summary()
        
        self.classification_loss = tf.keras.losses.SparseCategoricalCrossentropy()
            
        self.metrics_acc = tf.keras.metrics.Accuracy()

        if decay_rate != 0.0:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)
        else:
            lr = learning_rate

        self.optimizer = tf.keras.optimizers.Adam(lr)
        
        self.metrics_validation = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.metrics_acc_validation = tf.keras.metrics.Accuracy()
        self.global_validation_metric = float("inf")
        self.global_validation_step = args.in_epoch

    def make_lstm_network(self):
        model = tf.keras.Sequential()
        
        model.add(layers.InputLayer(input_shape=[None, self.num_channels]))
        model.add(layers.GRU(units=2048, activation='tanh', use_bias=False, name="block15_gru"))

        model.add(layers.Dense(self.num_classes, activation='softmax', name='predictions'))

        return model
        
        # model.add(layers.GRU(units=2048, activation='tanh', use_bias=False, name="block15_gru", return_sequences=True))
        # model.add(layers.GlobalAveragePooling1D())

        # model = tf.keras.Sequential()
        # model.add(layers.InputLayer(input_shape=[None, self.num_channels]))
        # model.add(layers.GRU(units=1024, activation='tanh', use_bias=False, name="block15_gru", return_sequences=True))
        # model.add(layers.GlobalMaxPooling1D())
        
        # model.add(layers.Dense(self.num_classes, activation='softmax', name='predictions'))

        # return model

        # x0 = tf.keras.Input(shape=[None, self.num_channels])

        # x, st = layers.GRU(units=self.num_channels, activation='tanh', use_bias=False, name="block15_gru", return_sequences=True, return_state=True)(x0)
        # x, st = layers.GRU(units=self.num_channels, activation='tanh', use_bias=False, name="block16_gru", return_sequences=True, return_state=True)(x, initial_state=st)
        # x = layers.GRU(units=self.num_channels, activation='tanh', use_bias=False, name="block17_gru")(x, initial_state=st)

        # x = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # return tf.keras.Model(inputs=x0, outputs=x)




    @tf.function
    def train_step(self, train_tuple):
        
        images = train_tuple[0]
        labels = train_tuple[self.enumerate_index]
        sample_weight = None

        if self.class_weights_index != -1:
            sample_weight = train_tuple[self.class_weights_index]

        with tf.GradientTape() as tape:
            
            images = tf.map_fn(lambda x: tf.random.shuffle(x), images)
            x_c = self.lstm_class(images, training=True)
            loss = self.classification_loss(labels, x_c, sample_weight=sample_weight)

            var_list = self.trainable_variables

            gradients = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(gradients, var_list))

            return loss, x_c

    def valid_step(self, dataset_validation):

        for valid_tuple in dataset_validation:
            images = valid_tuple[0]
            labels = valid_tuple[self.enumerate_index]
            
            x_c = self.lstm_class(images, training=False)
            
            self.metrics_validation.update_state(labels, x_c)

            prediction = tf.argmax(x_c, axis=1)
            self.metrics_acc_validation.update_state(labels, prediction)

        validation_result = self.metrics_validation.result()
        acc_result = self.metrics_acc_validation.result()
        tf.summary.scalar('validation_loss', validation_result, step=self.global_validation_step)
        tf.summary.scalar('validation_acc', acc_result, step=self.global_validation_step)
        self.global_validation_step += 1

        print("validation loss:", validation_result.numpy(), "acc:", acc_result.numpy())
        improved = False
        if validation_result < self.global_validation_metric:
            self.global_validation_metric = validation_result
            improved = True

        return improved

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(
            lstm_class=self.lstm_class,
            optimizer=self.optimizer)

    def summary(self, images, tr_step, step):
        labels = tf.reshape(images[1], [-1])

        loss = tr_step[0]
        prediction = tf.argmax(tr_step[1], axis=1)

        self.metrics_acc.update_state(labels, prediction)
        acc_result = self.metrics_acc.result()

        print("step", step, "loss", loss.numpy(), "acc", acc_result.numpy())
        print(labels.numpy())
        print(prediction.numpy())
        
        tf.summary.image('features', tf.reshape(images[0], tf.shape(images[0]).numpy().tolist() + [1])/255, step=step)
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('accuracy', acc_result, step=step)

    def save_model(self, save_model):
        self.lstm_class.summary()
        self.lstm_class.save(save_model)