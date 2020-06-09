
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import os

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
        self.reg_constant = 1e-3
        self.drop_prob = drop_prob
        
        # self.projection = self.make_projection_model()
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.encoder = self.make_encoder_model()
        
        self.generator.summary()
        self.discriminator.summary()
        self.encoder.summary()

        lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)
        

        self.encoder_optimizer = tf.keras.optimizers.Adam(lr)
        self.generator_optimizer = tf.keras.optimizers.Adam(lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr)
        
        # self.step = tf.Variable(0, dtype=tf.int64)
        # self.skip_steps = 5
        
    def set_nn2(self, ed_nn):
        ed_nn.trainable = False
        self.ed_nn = ed_nn

    # def make_projection_model(self):

    #     model = tf.keras.Sequential()

    #     model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(4096,), kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
    #     model.add(layers.BatchNormalization())
    #     model.add(layers.LeakyReLU())
    #     model.add(layers.Reshape((4, 4, 1024)))

    #     return model


    def make_generator_model(self):

        model = tf.keras.Sequential()

        model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(4096,), kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((4, 4, 1024)))

        model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.num_channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01), name="block7"))

        return model

    def make_discriminator_model(self):

        model = tf.keras.Sequential()
        

        model.add(layers.BatchNormalization(input_shape=[512, 512, self.num_channels]))
        model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2)))

        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2)))

        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2)))

        model.add(layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2)))

        model.add(layers.Reshape((4*4*1024,)))
        model.add(layers.Dense(1, use_bias=False))

        return model

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
    def GeodesicDistance(self, x, y):
        return tf.math.acos(tf.reduce_sum(tf.multiply(x, y)))

    @tf.function
    def GeodesicDistances(self, x_e):
        batch_size = tf.shape(x_e)[0]
        return tf.reduce_sum(tf.map_fn(lambda i: tf.reduce_sum(tf.map_fn(lambda j: self.GeodesicDistance(x_e[i], x_e[j]), tf.range(i + 1, batch_size), dtype=tf.float32)), tf.range(batch_size), dtype=tf.float32))

    @tf.function
    def train_step(self, images):
        images = images[1]/255
        batch_size = tf.shape(images)[0]
        noise = tf.math.abs(tf.linalg.normalize(tf.random.normal([batch_size, 4096]), axis=1)[0])
        # self.step.assign_add(1)

        with tf.GradientTape() as enc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            x_e = tf.math.abs(tf.linalg.normalize(self.encoder(images, training=True), axis=1)[0])
            x_logit = self.generator(x_e, training=True)

            enc_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=images), axis=[1, 2, 3]))

            generated_images = tf.nn.sigmoid(self.generator(noise, training=True))

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            fake_output_x = self.discriminator(tf.nn.sigmoid(x_logit), training=True)
            
            # gen_loss = self.generator_loss(fake_output) + enc_loss*(tf.cast(self.step % self.skip_steps < 1, dtype=tf.float32))*1e-7
            gen_loss = self.generator_loss(fake_output) + self.generator_loss(fake_output_x) + enc_loss*self.reg_constant
            
            # enc_loss -= self.GeodesicDistances(x_e)

            disc_loss = self.discriminator_loss(real_output, fake_output)
        

        encoder_train_variables = self.encoder.trainable_variables
        generator_train_variables = self.generator.trainable_variables
        discriminator_train_variables = self.discriminator.trainable_variables

        gradients_of_encoder = enc_tape.gradient(enc_loss, encoder_train_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator_train_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_train_variables)

        self.encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder_train_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_train_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_train_variables))

        return gen_loss, disc_loss, enc_loss, generated_images, x_logit

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output) 

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(encoder=self.encoder,
            discriminator=self.discriminator,
            generator=self.generator, 
            encoder_optimizer=self.encoder_optimizer,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer)

    def save_model(self, save_model):
        # self.encoder.save(save_model)
        self.generator.save(save_model)


    def summary(self, images, tr_strep, step):
        
        loss_g = tr_strep[0]
        loss_d = tr_strep[1]
        enc_loss = tr_strep[2]
        generated_images = tr_strep[3]
        x_logit = tf.nn.sigmoid(tr_strep[4])

        print("step", step, "loss_g", loss_g.numpy(), "loss_d", loss_d.numpy(), "enc_loss", enc_loss.numpy())
        
        tf.summary.scalar('enc_loss', enc_loss, step=step)
        tf.summary.scalar('loss_g', loss_g, step=step)
        tf.summary.scalar('loss_d', loss_d, step=step)
        tf.summary.image('real', images[1]/255, step=step)
        tf.summary.image('generated', generated_images, step=step)
        tf.summary.image('generated_x', x_logit, step=step)
    
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
        