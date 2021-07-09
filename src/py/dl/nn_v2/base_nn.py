from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import glob
import sys

class BaseNN(tf.keras.Model):

    def __init__(self, data_description=None):
        super(BaseNN, self).__init__()
        self.data_description = {}
        self.keys_to_features = {}
        self.json_filename = ""
        self.data_description = {}
        self.keys_to_features = {}
        self.num_channels = 1
        self.out_channels = 1
        self.global_step = 0
        self.set_data_description(data_description)

    def set_global_step(self, step):
        self.global_step = step

    def get_global_step(self):
        return self.global_step
    
    def read_and_decode(self, record):

        parsed = tf.io.parse_single_example(record, self.keys_to_features)
        reshaped_parsed = []

        if("data_keys" in self.data_description):
            for data_key in self.data_description["data_keys"]:
                reshaped_parsed.append(tf.reshape(parsed[data_key], self.data_description[data_key]["shape"]))

            return tuple(reshaped_parsed)

        return parsed

    def tf_inputs(self, batch_size=1, buffer_size=1000):

        tfrecords_arr = []
        tfrecords_dir = os.path.join(os.path.dirname(self.json_filename), self.data_description["tfrecords"], '**/*.tfrecord')
        for tfr in glob.iglob(tfrecords_dir, recursive=True):
          tfrecords_arr.append(tfr)

        dataset = tf.data.TFRecordDataset(tfrecords_arr)
        dataset = dataset.map(self.read_and_decode)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)

        return dataset

    def callbacks(self, checkpoint_path, summary_path):
        return [
        tf.keras.callbacks.TensorBoard(
            log_dir=summary_path, histogram_freq=0, write_graph=True, write_images=True,
            update_freq=100, profile_batch=0, embeddings_freq=0,
            embeddings_metadata=None
            ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='accuracy', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch'
            )
        ]
