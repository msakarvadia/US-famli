
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import os

class NN(tf.keras.Model):

    def __init__(self, learning_rate, decay_steps, decay_rate, staircase):
        super(NN, self).__init__()
        
        self.num_channels = 3

        self.encoder = self.make_encoder_model()
        self.generator = self.make_generator_model()

        self.encoder.summary()
        self.generator.summary()

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        
    def set_nn2(self, ed_nn):
        ed_nn.trainable = False
        self.ed_nn = ed_nn

    def make_generator_model(self):

        model = tf.keras.Sequential()
        model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(4096,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((4, 4, 1024)))

        return model

    def make_encoder_model(self):

        model = tf.keras.Sequential()
        
        model.add(layers.Reshape((4*4*1024,), input_shape=(4*4*1024,)))
        model.add(layers.Dense(4096, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        return model

    @tf.function
    def train_step(self, images):
        train_images = images[1]/255

        with tf.GradientTape() as tape:

            x_e = self.encode(self.ed_nn.encode(train_images))
            x_logit = self.ed_nn.decode(self.decode(x_e))

            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=train_images))
            batch_size = tf.shape(x_e)[0]

            loss += self.emd(x_e, tf.random.normal([batch_size, 4096]))
            var_list = self.trainable_variables
            gradients = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(gradients, var_list))

            return loss, x_logit

    def encode(self, x):
        return self.encoder(x)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generator(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(encoder=self.encoder,
            generator=self.generator, 
            optimizer=self.optimizer)

    def summary(self, images, tr_step, step):

        loss = tr_step[0]
        x_logit = tr_step[1]
        print("loss", loss.numpy())
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.image('real', images[1]/255, step=step)
        tf.summary.image('generated', tf.sigmoid(x_logit), step=step)

    
    def histogram(self, x, y, nbins=100, range_h=None):
        
        shape = tf.shape(y)
        batch_size = shape[0]
        
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        
        range_h = [tf.reduce_min(tf.concat([x, y], axis=-1)), tf.reduce_max(tf.concat([x, y], axis=-1))]
        
        # hisy_bins is a Tensor holding the indices of the binned values whose shape matches y.
        histy_bins = tf.histogram_fixed_width_bins(y, range_h, nbins=nbins, dtype=tf.int32)
        # and creates a histogram_fixed_width 
        H = tf.map_fn(lambda i: tf.histogram_fixed_width(tf.boolean_mask(x, tf.equal(histy_bins, i)), range_h, nbins=nbins, dtype=tf.int32), tf.range(nbins))
        
        return tf.cast(H, dtype=tf.float32)
    
    @tf.function
    def emd(self, x, y, nbins=255):

        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        range_h = [tf.reduce_min(tf.concat([x, y], axis=-1)), tf.reduce_max(tf.concat([x, y], axis=-1))]

        histo_x = tf.cast(tf.histogram_fixed_width(x, range_h, nbins=nbins, dtype=tf.int32), dtype=tf.float32)
        histo_y = tf.cast(tf.histogram_fixed_width(y, range_h, nbins=nbins, dtype=tf.int32), dtype=tf.float32)

        all_sorted_xy = tf.sort(tf.concat([histo_x, histo_y], axis=-1))
        
        all_sorted_xy_delta = tf.cast(all_sorted_xy[1:] - all_sorted_xy[:-1], dtype=tf.float32)

        histo_x_sorted = tf.sort(histo_x)
        histo_y_sorted = tf.sort(histo_y)

        histo_x_indices = tf.searchsorted(histo_x_sorted, all_sorted_xy[:-1], side='right')
        histo_y_indices = tf.searchsorted(histo_y_sorted, all_sorted_xy[:-1], side='right')

        cmdf_x = tf.cast(tf.math.divide(histo_x_indices, nbins), dtype=tf.float32)
        cmdf_y = tf.cast(tf.math.divide(histo_y_indices, nbins), dtype=tf.float32)
        
        return tf.math.sqrt(tf.reduce_sum(tf.math.multiply(tf.math.squared_difference(cmdf_x, cmdf_y), all_sorted_xy_delta)))
        