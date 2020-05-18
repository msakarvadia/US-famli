
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import os

class NN(tf.keras.Model):

    def __init__(self, tf_inputs, learning_rate = 1e-4, decay_steps = 10000, decay_rate = 0.96, staircase = 0, drop_prob = 0):
        super(NN, self).__init__()
        
        data_description = tf_inputs.get_data_description()
        
        self.num_channels = data_description[data_description["data_keys"][1]]["shape"][-1]

        self.encoder = self.make_encoder_model()
        self.encoder.summary()

        lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)

        self.optimizer = tf.keras.optimizers.Adam(lr)
        
    def set_nn2(self, gan_ed_nn):
        gan_ed_nn.trainable = False
        self.gan_ed_nn = gan_ed_nn

    def make_encoder_model(self):

        model = tf.keras.Sequential()

        model.add(layers.BatchNormalization(input_shape=[512, 512, self.num_channels]))

        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((4*4*1024,)))
        model.add(layers.Dense(4096, use_bias=False))

        return model

    @tf.function
    def train_step(self, images):
        train_images = images[1]/255

        with tf.GradientTape() as tape:

            x_e = self.encode(train_images)
            x_logit = self.gan_ed_nn.generator(x_e, training=False)

            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=train_images))
            batch_size = tf.shape(x_e)[0]
            
            var_list = self.trainable_variables
            gradients = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(gradients, var_list))

            return loss, x_logit

    def encode(self, x):
        x_e = tf.linalg.normalize(self.encoder(x), axis=1)[0]
        return x_e 

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(encoder=self.encoder, 
            optimizer=self.optimizer)

    def summary(self, images, tr_step, step):

        loss = tr_step[0]
        x_logit = tr_step[1]
        print("step", step, "loss", loss.numpy())
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.image('real', images[1]/255, step=step)
        tf.summary.image('generated', tf.sigmoid(x_logit), step=step)
