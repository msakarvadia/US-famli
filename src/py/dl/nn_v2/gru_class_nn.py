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

class Attention(layers.Layer):
    def __init__(self, units, name='attention'):
        super(Attention, self).__init__()

        self.projection = layers.Dense(units, activation='tanh', use_bias=False, name=name + "_projection")

        self.wt = layers.Dense(units, activation=None, use_bias=False, name=name + "_wt")
        self.wx = layers.Dense(units, activation=None, use_bias=False, name=name + "_wx")

        self.add = layers.Add()
        self.add_act = layers.Activation('tanh')

        self.wa = layers.Dense(1, activation=None, use_bias=False, name=name + "_wa")
        self.softmax = layers.Softmax(axis=1)

    def call(self, x0):

        x = self.projection(x0)

        attention_wt = self.wt(x)
        attention_wx = self.wx(x)

        attention = self.add([attention_wt, attention_wx])
        attention = self.add_act(attention)
        
        attention_wa = self.wa(attention)
        attention_scores = self.softmax(attention_wa)

        return tf.reduce_sum(tf.math.multiply(x, attention_scores), axis=1)

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    # self.k = k

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]]), k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)
    # min_score = tf.reshape(min_score, [-1, 1, 1])
    # score_mask = tf.greater_equal(score, min_score)
    # score_mask = tf.cast(score_mask, tf.float32)
    # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

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

        self.gru_class = self.make_gru_network()
        self.gru_class.summary()
        
        # self.classification_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        label_smoothing = 0
        # self.classification_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        self.classification_loss = tf.keras.losses.BinaryCrossentropy()
            
        self.metrics_acc = tf.keras.metrics.Accuracy()

        if decay_rate != 0.0:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)
        else:
            lr = learning_rate

        self.optimizer = tf.keras.optimizers.Adam(lr)
        
        # self.metrics_validation = tf.keras.metrics.SparseCategoricalCrossentropy()
        # self.metrics_validation = tf.keras.metrics.CategoricalCrossentropy(label_smoothing=label_smoothing)
        self.metrics_validation = tf.keras.metrics.BinaryCrossentropy()
        self.metrics_acc_validation = tf.keras.metrics.Accuracy()
        self.global_validation_metric = float("inf")
        self.global_validation_step = args.in_epoch

    def make_gru_network(self):

        x0 = tf.keras.Input(shape=[None, self.num_channels])

        x = layers.Masking(mask_value=-1.0)(x0)

        x = layers.GaussianNoise(0.1)(x)

        x = layers.BatchNormalization()(x)
        
        x_e, x_h_fwd, x_h_bwd = layers.Bidirectional(layers.GRU(units=512, activation='tanh', use_bias=False, kernel_initializer="glorot_normal", dropout=self.drop_prob, return_sequences=True, return_state=True), name="bi_gru")(x)
        x_e = layers.Dropout(self.drop_prob)(x_e)
        x_h_fwd = layers.Dropout(self.drop_prob)(x_h_fwd)
        x_h_bwd = layers.Dropout(self.drop_prob)(x_h_bwd)

        x_a_fwd, w_a_fwd = BahdanauAttention(1024)(x_h_fwd, x_e)
        x_a_bwd, w_a_bwd = BahdanauAttention(1024)(x_h_bwd, x_e)

        x = tf.concat([x_h_fwd, x_a_fwd, x_h_bwd, x_a_bwd], axis=-1)

        x = layers.Dense(self.num_classes, activation='softmax', use_bias=False, name='predictions')(x)

        return tf.keras.Model(inputs=x0, outputs=x)


    @tf.function(experimental_relax_shapes=True)
    def train_step(self, train_tuple):
        
        images = train_tuple[0]
        labels = train_tuple[self.enumerate_index]
        sample_weight = None

        if self.class_weights_index != -1:
            sample_weight = train_tuple[self.class_weights_index]

        with tf.GradientTape() as tape:
            
            # images = tf.map_fn(lambda x: tf.random.shuffle(x), images)
            x_c = self.gru_class(images, training=True)
            loss = self.classification_loss(tf.reshape(tf.one_hot(tf.cast(labels, tf.int32), self.num_classes), tf.shape(x_c)), x_c, sample_weight=sample_weight)

            var_list = self.trainable_variables

            gradients = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(gradients, var_list))

            return loss, x_c

    def valid_step(self, dataset_validation):

        for valid_tuple in dataset_validation:
            images = valid_tuple[0]
            labels = valid_tuple[self.enumerate_index]
            
            x_c = self.gru_class(images, training=False)

            # self.metrics_validation.update_state(labels, x_c)
            
            sample_weight = None
            if self.class_weights_index != -1:
                sample_weight = valid_tuple[self.class_weights_index]
            self.metrics_validation.update_state(tf.reshape(tf.one_hot(tf.cast(labels, tf.int32), self.num_classes), tf.shape(x_c)), x_c, sample_weight=sample_weight)

            prediction = tf.argmax(x_c, axis=1)
            self.metrics_acc_validation.update_state(labels, prediction, sample_weight=sample_weight)

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
            gru_class=self.gru_class,
            optimizer=self.optimizer)

    def summary(self, train_tuple, tr_step, step):
        
        sample_weight = None
        if self.class_weights_index != -1:
            sample_weight = train_tuple[self.class_weights_index]

        labels = tf.reshape(train_tuple[1], [-1])

        loss = tr_step[0]
        prediction = tf.argmax(tr_step[1], axis=1)

        self.metrics_acc.update_state(labels, prediction, sample_weight=sample_weight)
        acc_result = self.metrics_acc.result()

        print("step", step, "loss", loss.numpy(), "acc", acc_result.numpy())
        print(labels.numpy())
        print(prediction.numpy())
        
        tf.summary.image('features', tf.reshape(train_tuple[0], tf.shape(train_tuple[0]).numpy().tolist() + [1])/255, step=step)
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('accuracy', acc_result, step=step)

    def save_model(self, save_model):
        self.gru_class.summary()
        self.gru_class.save(save_model)