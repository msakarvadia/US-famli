
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

        data_description = tf_inputs.get_data_description()
        
        self.num_channels = data_description[data_description["data_keys"][1]]["shape"][-1]

        self.discriminator = self.make_discriminator_model()
        self.generator = self.make_generator_model()

        self.discriminator.summary()
        self.generator.summary()

        lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)

        self.generator_optimizer = tf.keras.optimizers.Adam(lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr)
        

    def make_generator_model(self):

        model = tf.keras.Sequential(name="generator_512")
        model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(4096,), kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name="block0"))

        model.add(layers.Reshape((4, 4, 1024)))

        model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name="block1"))

        model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name="block2"))

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name="block3"))

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name="block4"))

        model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name="block5"))

        model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(name="block6"))

        model.add(layers.Conv2DTranspose(self.num_channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01), name="block7"))

        return model 

    def make_discriminator_model(self):

        model = tf.keras.Sequential(name="discriminator_512")

        model.add(layers.BatchNormalization(input_shape=[512, 512, self.num_channels]))
        model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2), name="block0"))

        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2), name="block1"))

        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2), name="block2"))

        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2), name="block3"))

        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2), name="block4"))

        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2), name="block5"))

        model.add(layers.Conv2D(1024, (3, 3), strides=(1, 1), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.AveragePooling2D((2, 2), name="block6"))

        model.add(layers.Reshape((4*4*1024,)))
        model.add(layers.Dense(1, use_bias=False, name="block7"))

        return model

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)


    @tf.function
    def train_step(self, images):
        images = images[1]/255
        batch_size = tf.shape(images)[0]
        noise = tf.math.abs(tf.linalg.normalize(tf.random.normal([batch_size, 4096]), axis=1)[0])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = tf.nn.sigmoid(self.generator(noise, training=True))

            fake_output = self.discriminator(generated_images, training=True)
            real_output = self.discriminator(images, training=True)

            gen_loss = self.generator_loss(fake_output) + self.emd(generated_images, images)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))        

        return generated_images, gen_loss, disc_loss

    def summary(self, images, tr_step, step):

        images = images[1]
        batch_size = tf.shape(images)[0]
        
        generated_images = tr_step[0]

        gen_loss = tr_step[1]
        disc_loss = tr_step[2]

        tf.summary.scalar("gen_loss", gen_loss, step=step)
        tf.summary.scalar("disc_loss", disc_loss, step=step)
        tf.summary.image('generated', generated_images, step=step)
        tf.summary.image('real', images/255., step=step)

        print(step, "g_loss", gen_loss.numpy(), "d_loss", disc_loss.numpy())

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(generator=self.generator, 
            discriminator=self.discriminator, 
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer)

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
