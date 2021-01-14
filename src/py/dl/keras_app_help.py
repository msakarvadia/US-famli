import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import os
import json


# x0 = layers.Input(shape=[193, 256, 3])
x0 = layers.Input(shape=[256, 256, 3])

# model = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=x0, pooling='max')
# model.summary()
# model.save("/work/jprieto/data/remote/GWH/Users/jprieto/jprieto/data/dataset_c/trained/keras_app/InceptionResNetV2_extract_features_maxpool")

# model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=x0, pooling='avg')
# model.summary()
# model.save("/ASD/juan_flyby/CVPR_modelNet40/trained/keras_app/ResNet50_extract_features_avgpool")

# model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_tensor=x0, pooling=None)
# model.summary()
# model.save("/ASD/juan_flyby/CVPR_modelNet40/trained/keras_app/xception_extract_features_256_nonepool")

model = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=x0, pooling=None)
model.summary()
model.save("/path/to/output")

# model = tf.keras.models.load_model("/work/jprieto/data/remote/GWH/Users/jprieto/jprieto/data/dataset_c/trained/keras_app/vgg19_extract_features_avgpool_256")
# model.summary()
# model_out = tf.keras.models.Sequential([layers.InputLayer(input_shape=[386, 512, 3])] + model.layers[1:])
# model_out.summary()
# model_out.save("/work/jprieto/data/remote/GWH/Users/jprieto/jprieto/data/dataset_c/trained/keras_app/vgg19_extract_features_avgpool_512_386")




# model = tf.keras.models.load_model('/work/jprieto/data/remote/GWH/Users/jprieto/data/FAMLI_a_b/trained/UNCHCS_flat_ga_cleaned_dataset_a_b_c_BPD_only_split_train_split_train/vgg19_valid/')
# model.layers = model.layers[0:-4]
# model.layers.append(layers.GlobalAveragePooling2D())
# model.summary()