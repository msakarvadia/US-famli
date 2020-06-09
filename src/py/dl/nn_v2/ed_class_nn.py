
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
        self.num_channels = data_description[data_description["data_keys"][0]]["shape"][-1]
        
        self.num_classes = 2
        if(data_description[data_description["data_keys"][1]]["num_class"]):
            self.num_classes = data_description[data_description["data_keys"][1]]["num_class"]
            print("Number of classes in data description", self.num_classes)

        self.drop_prob = drop_prob
        
        self.classifier = self.make_classification_model()
        self.classification_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
        self.sample_weight = None
        if(sample_weight is not None):
            self.sample_weight = tf.constant(np.reshape(np.multiply(np.ones([args.batch_size, self.num_classes]), sample_weight), [args.batch_size, self.num_classes, 1]), dtype=tf.float32)
            print("Weights:", self.sample_weight.numpy())
            
        self.metrics_acc = tf.keras.metrics.Accuracy()
        
        self.classifier.summary()

        lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)

        self.optimizer = tf.keras.optimizers.Adam(lr)
        

    def set_nn2(self, ed_nn):
        self.encoder = ed_nn.encoder
        self.encoder.trainable = False


    def make_classification_model(self):

        model = tf.keras.Sequential()

        model.add(layers.Reshape((4096,), input_shape=(1, 1, 4096)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(self.num_classes, use_bias=False))

        return model

    @tf.function
    def train_step(self, train_tuple):
        
        images = train_tuple[0]/255
        labels = train_tuple[1]

        with tf.GradientTape() as tape:
            
            x_c = self.classifier(self.encoder(images))
            loss = self.classification_loss(tf.one_hot(labels, self.num_classes, axis=1), x_c, sample_weight=self.sample_weight)

            var_list = self.trainable_variables

            gradients = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(gradients, var_list))

            return loss, x_c

    def save_model(self, save_model):
        model = tf.keras.Sequential([layers.Lambda(lambda x: x/255, input_shape=(512, 512, self.num_channels))] + self.encoder.layers + self.classifier.layers)
        model.add(layers.Softmax())
        model.summary()
        model.save(save_model)

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(
            encoder=self.encoder,
            classifier=self.classifier,
            optimizer=self.optimizer)

    def summary(self, images, tr_step, step):
        labels = tf.reshape(images[1], -1)
        loss = tr_step[0]
        prediction = tf.argmax(tf.nn.softmax(tr_step[1]), axis=1)

        self.metrics_acc.update_state(labels, prediction)
        acc_result = self.metrics_acc.result()

        print("step", step, "loss", loss.numpy(), "acc", acc_result.numpy(), labels.numpy(), prediction.numpy())
        tf.summary.image('real', images[0]/255, step=step)
        tf.summary.scalar('accuracy', acc_result, step=step)