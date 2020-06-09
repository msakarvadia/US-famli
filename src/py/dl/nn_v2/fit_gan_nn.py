
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import os
import time

class FitGan(tf.keras.Model):

    def __init__(self, init_var, learning_rate = 1e-3):
        super(FitGan, self).__init__()

        self.fit_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.fit = tf.Variable(init_var, dtype=tf.float32)

    def set_nn2(self, gan_ed_nn):
        self.gan_ed_nn = gan_ed_nn
        self.gan_ed_nn.trainable = False

    @tf.function
    def train_step(self, images):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            x_logit = self.gan_ed_nn.generator(self.fit)

            loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=images), axis=[1, 2, 3]))
        
        gradients_of_fit = gen_tape.gradient(loss, [self.fit])

        self.fit_optimizer.apply_gradients(zip(gradients_of_fit, [self.fit]))

        return x_logit, loss


class NN():
    def __init__(self, tf_inputs, args):
        super(NN, self).__init__()

        learning_rate = args.learning_rate
        decay_steps = args.decay_steps
        decay_rate = args.decay_rate
        staircase = args.staircase
        drop_prob = args.drop_prob
        sample_weight = args.sample_weight

    def set_nn2(self, gan_ed_nn):
        self.gan_ed_nn = gan_ed_nn
        self.gan_ed_nn.trainable = False


    def train_step(self, images):
        images = images[1]/255
        batch_size = tf.shape(images)[0]

        self.fit_gan = FitGan(tf.linalg.normalize(tf.random.normal(shape=(batch_size, 4096)))[0])
        self.fit_gan.set_nn2(self.gan_ed_nn)

        start = time.time()

        loss_min = 9999999
        did_not_improve = 0

        for step in range(100000):
            
            x_logit, loss = self.fit_gan.train_step(images)

            if(loss < loss_min):
                loss_min = loss
                x_logit_ = x_logit
                did_not_improve = 0
            else:
                did_not_improve += 1

            if(did_not_improve > 100):
                break

        print("Fitting took", time.time() - start)

        return x_logit_, loss_min
    

    def summary(self, images, tr_step, step):

        images = images[1]
        batch_size = tf.shape(images)[0]
        
        x_logit = tr_step[0]
        loss = tr_step[1]

        tf.summary.scalar("loss", loss, step=step)
        tf.summary.image('generated', tf.sigmoid(x_logit), step=step)
        tf.summary.image('real', images/255., step=step)

        print(step, "loss", loss.numpy())

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint()
