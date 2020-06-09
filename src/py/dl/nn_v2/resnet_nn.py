
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import os

class ConvBlock(layers.Layer):
    def __init__(self, out_filters):
        super(ConvBlock, self).__init__()

        self.conv1 = layers.Conv2D(out_filters[0], (1, 1), strides=(1, 1), use_bias=False, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.LeakyReLU()

        self.conv2 = layers.Conv2D(out_filters[1], (3, 3), strides=(2, 2), use_bias=False, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.LeakyReLU()

        self.conv3 = layers.Conv2D(out_filters[2], (1, 1), strides=(1, 1), use_bias=False, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.ac3 = layers.LeakyReLU()

        self.conv4 = layers.Conv2D(out_filters[2], (1, 1), strides=(2, 2), use_bias=False, padding='same')
        self.bn4 = layers.BatchNormalization()

        self.add = layers.Add()
        self.ac4 = layers.LeakyReLU()

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

class IdentityBlock(layers.Layer):
    def __init__(self, out_filters):
        super(IdentityBlock, self).__init__()

        self.conv1 = layers.Conv2D(out_filters[0], (1, 1), strides=(1, 1), use_bias=False, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.LeakyReLU()

        self.conv2 = layers.Conv2D(out_filters[1], (3, 3), strides=(1, 1), use_bias=False, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.LeakyReLU()

        self.conv3 = layers.Conv2D(out_filters[2], (1, 1), strides=(1, 1), use_bias=False, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.ac3 = layers.LeakyReLU()

        self.add = layers.Add()
        self.ac4 = layers.LeakyReLU()

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

        x = self.add([x, x0])

        return self.ac4(x)

class NN(tf.keras.Model):

    def __init__(self, tf_inputs, learning_rate = 1e-4, decay_steps = 10000, decay_rate = 0.96, staircase = 0, drop_prob = 0):
        super(NN, self).__init__()

        data_description = tf_inputs.get_data_description()
        
        self.num_channels = data_description[data_description["data_keys"][1]]["shape"][-1]

        self.num_classes = 2
        if(data_description[data_description["data_keys"][1]]["num_class"]):
            self.num_classes = data_description[data_description["data_keys"][1]]["num_class"]
            print("Number of classes in data description", self.num_classes)

        self.resnet = self.make_resnet_model()
        self.classification = self.make_classification_model()
        
        self.resnet.summary()
        self.classification.summary()

        self.resnet_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)

        self.optimizer = tf.keras.optimizers.Adam(lr)

    def make_resnet_model(self):

        model = tf.keras.Sequential()

        model.add(layers.Conv2D(64, (1, 1), strides=(2, 2), input_shape=[512, 512, self.num_channels], use_bias=False, padding='same'))

        model.add(ConvBlock([64,64,128]))
        model.add(IdentityBlock([64,64,128]))

        model.add(ConvBlock([64,64,128]))
        model.add(IdentityBlock([64,64,128]))

        model.add(ConvBlock([128,128,256]))
        model.add(IdentityBlock([128,128,256]))
        model.add(IdentityBlock([128,128,256]))

        model.add(ConvBlock([128,128,256]))
        model.add(IdentityBlock([128,128,256]))
        model.add(IdentityBlock([128,128,256]))

        model.add(ConvBlock([256,256,512]))
        model.add(IdentityBlock([256,256,512]))
        model.add(IdentityBlock([256,256,512]))
        model.add(IdentityBlock([256,256,512]))

        model.add(ConvBlock([512,512,1024]))
        model.add(IdentityBlock([512,512,1024]))
        model.add(IdentityBlock([512,512,1024]))
        model.add(IdentityBlock([512,512,1024]))
        model.add(IdentityBlock([512,512,1024]))

        return model

    def make_classification_model(self):

        model = tf.keras.Sequential()

        model.add(layers.BatchNormalization(input_shape=(4, 4, 1024)))
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((4*4*1024,)))
        model.add(layers.Dense(self.num_classes, use_bias=False))

        return model


    @tf.function
    def train_step(self, images):
        labels = images[1]
        images = images[0]/255

        with tf.GradientTape() as resnet_tape:

            x_logits = self.classification(self.resnet(images))

            loss = self.resnet_loss(labels, x_logits)

        gradients_of_resnet = resnet_tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients_of_resnet, self.trainable_variables))

        return loss

    def summary(self, images, tr_step, step):

        images = images[0]
        loss = tr_step

        tf.summary.scalar("loss", loss, step=step)
        tf.summary.image('images', images/255., step=step)

        print(step, "loss", loss.numpy())

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(resnet=self.resnet,
            classification=self.classification,
            optimizer=self.optimizer)
