import numpy as np
import os
import glob
import sys
import json
import tensorflow as tf

from tensorflow import feature_column
from pandas import read_csv, notna, notnull


class TFInputs():

    def __init__(self, json_filename):

        self.json_filename = json_filename
        self.keys_to_features = {}

        with open(json_filename, "r") as f:
            self.data_description = json.load(f)

        if("data_keys" in self.data_description):
            for data_key in self.data_description["data_keys"]:
                self.keys_to_features[data_key] = tf.io.FixedLenFeature((np.prod(self.data_description[data_key]["shape"])), eval(self.data_description[data_key]["type"]))

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

        if "csv" in self.data_description:
            return self.tf_inputs_dataframe(batch_size, buffer_size)
        else:
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


    def tf_inputs_dataframe(self, batch_size=1, buffer_size=1000):
        dataframe = read_csv(os.path.join(os.path.dirname(self.json_filename), self.data_description["csv"]))
        labels_name = 'ga_edd'
        y_name = labels_name

        for column_name in dataframe.columns:
          if column_name.startswith('_'):
            dataframe.pop(column_name)
        
        for header in ['fl_1', 'bp_1', 'hc_1', 'ac_1', 'mom_age_edd', 'mom_weight_lb', 'mom_height_in']:
          r = max(dataframe[header]) - min(dataframe[header])
          dataframe[header] = (dataframe[header] - min(dataframe[header]))/r

        dataframe = dataframe[ (dataframe[y_name] != '.') & (notna(dataframe[y_name])) & (notnull(dataframe[y_name]))].copy()
        dataframe = dataframe.astype( {y_name : 'int32'})
        
        feature_columns = []
        feature_names = []
        num_channels = 0
        for header in ['fl_1', 'bp_1', 'hc_1', 'ac_1', 'mom_age_edd', 'mom_weight_lb', 'mom_height_in']:
            feature_columns.append(feature_column.numeric_column(header))
            feature_names.append(header)
            num_channels += 1

        num_identity = 2
        for header in ['hiv', 'current_smoker', 'former_smoker', 'chronic_htn', 'preg_induced_htn', 'diabetes', 'gest_diabetes']:
            col = feature_column.categorical_column_with_identity(header, num_identity)
            col = feature_column.indicator_column(col)
            feature_columns.append(col)
            feature_names.append(header)
            num_channels += num_identity

        self.num_channels = num_channels

        feature_layer = tf.keras.layers.DenseFeatures(feature_columns=feature_columns)
        dataframe = dataframe.copy()
        labels = dataframe.pop(labels_name)

        dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda x, y: (feature_layer(x), y))

        return dataset
