
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

        self.drop_prob = drop_prob

        self.encoder = self.make_encoder_model()
        self.generator = self.make_generator_model()

        self.gaussian = tf.keras.layers.GaussianNoise(1.0)

        self.encoder.summary()
        self.generator.summary()

        lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)

        self.optimizer = tf.keras.optimizers.Adam(lr)
        

    def make_generator_model(self):

        model = tf.keras.Sequential()

        model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(4096,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((4, 4, 1024)))
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2DTranspose(self.num_channels, (3, 3), strides=(2, 2), padding='same', use_bias=False))

        return model

    def make_encoder_model(self):

        model = tf.keras.Sequential()

        model.add(layers.BatchNormalization(input_shape=[512, 512, self.num_channels]))

        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.drop_prob))

        model.add(layers.Reshape((4*4*1024,)))
        model.add(layers.Dense(4096, use_bias=False))

        return model

    @tf.function
    def train_step(self, images):
        
        noise_images = self.gaussian(images[1])
        noise_images = tf.divide(tf.subtract(noise_images, tf.reduce_min(noise_images)), tf.subtract(tf.reduce_max(noise_images), tf.reduce_min(noise_images)))

        with tf.GradientTape() as tape:
            
            x_e = self.encode(noise_images)
            x_logit = self.decode(x_e)

            loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=noise_images), axis=[1, 2, 3]))
            
            var_list = self.trainable_variables
            
            gradients = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(gradients, var_list))

            return loss, x_logit

    @tf.function
    def encode(self, images, mean=0., sigma=1.):
        x_e = self.encoder(images)
        return tf.math.abs(tf.linalg.normalize(x_e, axis=1)[0])

    def save_model(self, save_model):
        self.encoder.save(save_model)
        # self.generator.save(save_model)
        
    def sample(self):
        x_e = tf.linalg.normalize(tf.random.normal(shape=(3, 4096)), axis=1)[0]
        x_e = tf.math.abs(x_e)
        return self.decode(x_e, apply_sigmoid=True)

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
        tf.summary.image('sampled', self.sample(), step=step)
    
    
    def mutual_info_histo(self, hist2d):
        # Get probability
        pxy = tf.divide(hist2d, tf.reduce_sum(hist2d))
        # marginal for x over y
        px = tf.reduce_sum(pxy, axis=1)
        # marginal for y over x
        py = tf.reduce_sum(pxy, axis=0)

        px_py = tf.multiply(px[:, None], py[None, :])

        px_py = tf.boolean_mask(px_py, pxy)
        pxy = tf.boolean_mask(pxy, pxy)

        return tf.reduce_sum(tf.multiply(pxy, tf.math.log(tf.math.divide(pxy, px_py))))

    
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

    
    def mutual_info(self, x, y, nbins=255):
        return self.mutual_info_histo(self.histogram(x, y))

    
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
        