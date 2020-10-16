
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import os
import json
import math

class IoU(tf.keras.losses.Loss):
    def __init__(self, one_hot=True, num_classes=1):
        super(IoU, self).__init__()

        self.one_hot = one_hot
        self.num_classes = num_classes

    def call(self, y_true, y_pred):

        shape = tf.shape(y_pred)
        batch_size = shape[0]

        y_pred_flat = tf.reshape(y_pred, [batch_size, -1])

        if self.one_hot:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes)

        y_true_flat = tf.reshape(tf.cast(y_true, tf.float32), [batch_size, -1])

        intersection = 2.0 * tf.reduce_sum(y_pred_flat * y_true_flat, axis=1)
        denominator = tf.reduce_sum(y_pred_flat, axis=1) + tf.reduce_sum(y_true_flat, axis=1)

        return 1.0 - tf.reduce_mean(intersection / denominator)
    

class UBlock(layers.Layer):
    def __init__(self, filters, name="ublock"):
        super(UBlock, self).__init__()

        self.conv1 = layers.Conv2D(filters[0], (1, 1), strides=(1, 1), use_bias=False, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.ReLU()

        self.conv2 = layers.SeparableConv2D(filters[1], (3, 3), strides=(1, 1), use_bias=False, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.ReLU()

        self.conv3 = layers.Conv2D(filters[2], (1, 1), strides=(1, 1), use_bias=False, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.ac3 = layers.ReLU()

        self.add = layers.Add()
        self.ac4 = layers.ReLU()

        self.conv4 = layers.Conv2D(filters[2], (1, 1), strides=(1, 1), use_bias=False, padding='same')
        self.bn4 = layers.BatchNormalization()

    def call(self, x0):
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.ac1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.ac3(x)

        shortcut = self.conv4(x0)
        shortcut = self.bn4(shortcut)

        x = self.add([x, shortcut])

        return self.ac4(x)

class NN(tf.keras.Model):

    def __init__(self, tf_inputs, args):
        super(NN, self).__init__()

        learning_rate = args.learning_rate
        decay_steps = args.decay_steps
        decay_rate = args.decay_rate
        staircase = args.staircase
        drop_prob = args.drop_prob

        self.drop_prob = drop_prob
        self.data_description = tf_inputs.get_data_description()
        data_description = self.data_description
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

        self.u_seg = self.make_seg_model()
        self.u_seg.summary()
        
        # self.loss = IoU(data_description[data_description["data_keys"][self.enumerate_index]]["shape"][-1] == 1, self.num_classes)
        self.loss = tf.keras.losses.BinaryCrossentropy()

        if decay_rate > 0:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)
        else:
            lr = learning_rate

        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.metrics_acc = tf.keras.metrics.Accuracy()

        self.metrics_validation = tf.keras.metrics.MeanAbsoluteError()
        self.global_validation_metric = float("inf")
        self.global_validation_step = args.in_epoch

    def make_seg_model(self):

        x0 = tf.keras.Input(shape=[128, 128, self.num_channels])

        d0 = UBlock([16, 16, 64], name="d0")(x0)
        x = layers.MaxPool2D((2, 2))(d0)

        d1 = UBlock([32, 32, 128], name="d1")(x)
        x = layers.MaxPool2D((2, 2))(d1)

        d2 = UBlock([64, 64, 256], name="d2")(x)
        x = layers.MaxPool2D((2, 2))(d2)

        d3 = UBlock([128, 128, 512], name="d3")(x)
        x = layers.MaxPool2D((2, 2))(d3)

        
        x = UBlock([256, 256, 1024], name="du")(x)


        x = layers.UpSampling2D(size=(2, 2))(x)
        x = tf.concat([x, d3], axis=-1)
        x = UBlock([128, 128, 512], name="u3")(x)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = tf.concat([x, d2], axis=-1)
        x = UBlock([64, 64, 256], name="u2")(x)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = tf.concat([x, d1], axis=-1)
        x = UBlock([32, 32, 128], name="u1")(x)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = tf.concat([x, d0], axis=-1)
        x = UBlock([16, 16, 64], name="u0")(x)

        x = layers.Conv2D(self.num_channels, (1, 1), strides=(1, 1), padding='same', activation='softmax', use_bias=False)(x)

        return tf.keras.Model(inputs=x0, outputs=x)


    @tf.function
    def train_step(self, train_tuple):
        
        images = train_tuple[0]
        target = train_tuple[1]

        with tf.GradientTape() as seg_tape:

            x_logits = self.u_seg(images)
            loss = self.loss(target, x_logits)

        gradients_of_seg = seg_tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients_of_seg, self.trainable_variables))

        return loss, x_logits

    def valid_step(self, dataset_validation):

        for valid_tuple in dataset_validation:
            images = valid_tuple[0]
            targets = valid_tuple[1]
            
            synths = self.u_seg(images, training=False)
            
            self.metrics_validation.update_state(targets, synths)

        validation_result = self.metrics_validation.result()
        tf.summary.scalar('validation_loss', validation_result, step=self.global_validation_step)
        self.global_validation_step += 1

        print("validation loss:", validation_result.numpy())
        if validation_result < self.global_validation_metric:
            self.global_validation_metric = validation_result
            return True

        return False

    def summary(self, images, tr_step, step):


        imgs = images[0]
        targets = images[1]

        loss = tr_step[0]
        synths = tr_step[1]

        self.metrics_acc.update_state(targets, synths)
        acc_result = self.metrics_acc.result()

        tf.summary.scalar("loss", loss, step=step)
        tf.summary.scalar('acc', acc_result, step=step)
        tf.summary.image('images', imgs/255.0, step=step)
        tf.summary.image('targets', targets, step=step)
        tf.summary.image('synths', synths, step=step)

        print(step, "loss", loss.numpy(), "acc", acc_result.numpy())

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(u_seg=self.u_seg,
            optimizer=self.optimizer)

    def save_model(self, save_model):
        model = self.u_seg
        model.summary()
        model.save(save_model)
