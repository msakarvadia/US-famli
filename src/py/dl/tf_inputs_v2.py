import numpy as np
import os
import glob
import sys
import json
import tensorflow as tf

class TFInputs():

    def __init__(self, json_filename):

        self.json_filename = json_filename
        self.keys_to_features = {}

        with open(json_filename, "r") as f:
            self.data_description = json.load(f)

        if("data_keys" in self.data_description):
            for data_key in self.data_description["data_keys"]:
                self.keys_to_features[data_key] = tf.io.FixedLenFeature((np.prod(self.data_description[data_key]["shape"])), eval(self.data_description[data_key]["type"]))
        else:
            print("Nothing to decode! data_keys missing in object description object. tfRecords.py creates this descriptor.")
            raise

        if("enumerate" in self.data_description and "num_class" in self.data_description[self.data_description["enumerate"]]):
            self.enumerate = self.data_description["enumerate"]
            self.num_classes = self.data_description[self.enumerate]["num_class"]
        else:
            self.num_classes = 2


    def get_data_description(self):
        return self.data_description

    def get_class_names(self):
        data_description = self.data_description
        if("enumerate" in data_description):
            class_dict = data_description[data_description["enumerate"]]["class"]
            class_obj = {}
            for key in class_dict:
              class_obj[class_dict[key]] = key
            return class_obj.values()
        return []

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
        print("Reading tfRecords from", tfrecords_dir)
        for tfr in glob.iglob(tfrecords_dir, recursive=True):
          tfrecords_arr.append(tfr)
        print("tfRecords found:", len(tfrecords_dir))
        
        dataset = tf.data.TFRecordDataset(tfrecords_arr)
        dataset = dataset.map(self.read_and_decode)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)

        return dataset