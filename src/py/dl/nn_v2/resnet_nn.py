
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import os
import json

def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return regularizers.l2(l2_weight_decay) if use_l2_regularizer else None

class ConvBlock(layers.Layer):
    def __init__(self, out_filters, use_l2_regularizer=True):
        super(ConvBlock, self).__init__()

        self.conv1 = layers.Conv2D(out_filters[0], (1, 1), strides=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer))
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.ReLU()

        self.conv2 = layers.Conv2D(out_filters[1], (3, 3), strides=(2, 2), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer))
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.ReLU()

        self.conv3 = layers.Conv2D(out_filters[2], (1, 1), strides=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer))
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(out_filters[2], (1, 1), strides=(2, 2), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer))
        self.bn4 = layers.BatchNormalization()

        self.add = layers.Add()
        self.ac3 = layers.ReLU()

    def call(self, x0):
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.ac1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.conv4(x0)
        shortcut = self.bn4(shortcut)

        x = self.add([x, shortcut])

        return self.ac3(x)

    def get_config(self):
        return {"conv1": self.conv1, "bn1": self.bn1, "conv2": self.conv2, "bn2": self.bn2, "conv3": self.conv3, "bn3": self.bn3, "conv4": self.conv4, "bn4": self.bn4}

class IdentityBlock(layers.Layer):
    def __init__(self, out_filters, use_l2_regularizer=True):
        super(IdentityBlock, self).__init__()

        self.conv1 = layers.Conv2D(out_filters[0], (1, 1), strides=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer))
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.ReLU()

        self.conv2 = layers.Conv2D(out_filters[1], (3, 3), strides=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer))
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.ReLU()

        self.conv3 = layers.Conv2D(out_filters[2], (1, 1), strides=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer))
        self.bn3 = layers.BatchNormalization()
        self.ac3 = layers.ReLU()

        self.add = layers.Add()
        self.ac4 = layers.ReLU()

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
        self.use_l2_regularizer = True

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

        self.resnet = self.make_resnet_model()
        self.resnet.summary()

        self.resnet_loss = tf.keras.losses.SparseCategoricalCrossentropy()

        lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)

        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.metrics_acc = tf.keras.metrics.Accuracy()

        self.metrics_validation = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.global_validation_metric = float("inf")
        self.global_validation_step = args.in_epoch

    def make_resnet_model(self):

        model = tf.keras.Sequential()

        model.add(layers.InputLayer(input_shape=[256, 256, self.num_channels]))

        model.add(layers.Lambda(lambda x: tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode='CONSTANT', constant_values=-2.0)))
        model.add(layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, padding='valid', kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(self.use_l2_regularizer)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        model.add(ConvBlock([64,64,256], self.use_l2_regularizer))
        model.add(IdentityBlock([64,64,256], self.use_l2_regularizer))
        model.add(IdentityBlock([64,64,256], self.use_l2_regularizer))

        model.add(ConvBlock([128,128,512], self.use_l2_regularizer))
        model.add(IdentityBlock([128,128,512], self.use_l2_regularizer))
        model.add(IdentityBlock([128,128,512], self.use_l2_regularizer))
        model.add(IdentityBlock([128,128,512], self.use_l2_regularizer))

        model.add(ConvBlock([256, 256, 1024], self.use_l2_regularizer))
        model.add(IdentityBlock([256, 256, 1024], self.use_l2_regularizer))
        model.add(IdentityBlock([256, 256, 1024], self.use_l2_regularizer))
        model.add(IdentityBlock([256, 256, 1024], self.use_l2_regularizer))
        model.add(IdentityBlock([256, 256, 1024], self.use_l2_regularizer))
        model.add(IdentityBlock([256, 256, 1024], self.use_l2_regularizer))

        model.add(ConvBlock([512,512,2048], self.use_l2_regularizer))
        model.add(IdentityBlock([512,512,2048], self.use_l2_regularizer))
        model.add(IdentityBlock([512,512,2048], self.use_l2_regularizer))
        
        model.add(layers.GlobalAveragePooling2D())
        
        model.add(layers.Dense(self.num_classes, kernel_initializer=initializers.RandomNormal(stddev=0.01), kernel_regularizer=_gen_l2_regularizer(self.use_l2_regularizer), bias_regularizer=_gen_l2_regularizer(self.use_l2_regularizer)))
        model.add(layers.Softmax())

        return model


    @tf.function
    def train_step(self, train_tuple):
        
        images = train_tuple[0]
        labels = train_tuple[self.enumerate_index]
        sample_weight = None

        if self.class_weights_index != -1:
            sample_weight = train_tuple[self.class_weights_index]

        with tf.GradientTape() as resnet_tape:

            x_logits = self.resnet(images)
            loss = self.resnet_loss(labels, x_logits, sample_weight=sample_weight)

        gradients_of_resnet = resnet_tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients_of_resnet, self.trainable_variables))

        return loss, x_logits

    def valid_step(self, dataset_validation):

        for valid_tuple in dataset_validation:
            images = valid_tuple[0]
            labels = valid_tuple[self.enumerate_index]
            
            x_c = self.resnet(images)
            
            self.metrics_validation.update_state(labels, x_c)

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
        labels = tf.reshape(images[self.enumerate_index], -1)

        loss = tr_step[0]
        prediction = tf.argmax(tr_step[1], axis=1)

        self.metrics_acc.update_state(labels, prediction)
        acc_result = self.metrics_acc.result()

        tf.summary.scalar("loss", loss, step=step)
        tf.summary.image('images', imgs, step=step)
        tf.summary.scalar('accuracy', acc_result, step=step)

        sample_weight = np.ones_like(labels.numpy())
        if self.class_weights_index != -1:
            sample_weight = np.reshape(images[self.class_weights_index].numpy(), -1)

        print(step, "loss", loss.numpy(), "acc", acc_result.numpy(), labels.numpy(), prediction.numpy(), sample_weight)

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(resnet=self.resnet,
            optimizer=self.optimizer)

    def save_model(self, save_model):
        model = self.resnet
        model.summary()
        model.save(save_model)

        # model = tf.keras.Sequential(self.classifier.layers)
        # model.add(layers.Softmax())
        # model.summary()
        # model.save(save_model)
