
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import u_nn as nn
# import conv_nn as nn
import os
from datetime import datetime
import json
import glob
import itk

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='U net segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--img', type=str, help='Image filename to filter with a trained model', required=True)
parser.add_argument('--out', type=str, help='Output image', default="out.nrrd")
parser.add_argument('--model', help='Input modelname', required=True)
parser.add_argument('--num_labels', help='Number of labels for the softmax output', type=int, default=2)
parser.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
parser.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

args = parser.parse_args()


image_name = args.img
model_name = args.model
out_name = args.out
num_labels = args.num_labels
ps_device = args.ps_device
w_device = args.w_device

print('image_name', image_name)
print('model_name', model_name)
print('ps_device', ps_device)
print('w_device', w_device)

InputType = itk.Image[itk.SS,2]
img_read = itk.ImageFileReader[InputType].New(FileName=image_name)
img_read.Update()
img = img_read.GetOutput()

img_np = itk.GetArrayViewFromImage(img)
img_shape = img_np.shape

# The image is 2d but we reshape according to the number of channels
if(len(img_shape) == 2):
    img_shape = img_shape + (1,)

img_shape = (1,) + img_shape

graph = tf.Graph()

with graph.as_default():

  x = tf.placeholder(tf.float32,shape=img_shape)
  
  keep_prob = tf.placeholder(tf.float32)

  y_conv = nn.inference(x, num_labels=num_labels, keep_prob=1.0, is_training=False, ps_device=ps_device, w_device=w_device)
  label = y_conv

  with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    # specify where to write the log files for import to TensorBoard
    now = datetime.now()

    label_map = sess.run([label], feed_dict={x: np.reshape(img_np, img_shape)})

    label_map = np.reshape(label_map[0], img_np.shape)

    img_np.setfield(label_map,img_np.dtype)

    print("Writing:", out_name)
    writer = itk.ImageFileWriter.New(FileName=out_name, Input=img)
    writer.Update()

        