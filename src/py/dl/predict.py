
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import itk

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='U net segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


in_group = parser.add_mutually_exclusive_group(required=True)
  
in_group.add_argument('--img', type=str, help='Input image for prediction')
in_group.add_argument('--dir', type=str, help='Directory with images for prediction')

parser.add_argument('--out', type=str, help='Output image or directory. If dir flag is used the output image name will be the <Directory set in out flag>/<imgage filename in directory dir>', default="./out.nrrd")
parser.add_argument('--model', help='Input modelname', required=True)
parser.add_argument('--nn', type=str, help='Type of neural network to use', default='u_nn')
parser.add_argument('--num_labels', help='Number of labels for the softmax output', type=int, default=2)
parser.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
parser.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

args = parser.parse_args()


model_name = args.model
neural_network = args.nn
out_name = args.out
num_labels = args.num_labels
ps_device = args.ps_device
w_device = args.w_device

print('neural_network', neural_network)
print('model_name', model_name)
print('neural_network', neural_network)
is_gan = "gan" in neural_network
if is_gan:
  print('Using gan scheme')
print('ps_device', ps_device)
print('w_device', w_device)
print('out', out_name)

nn = importlib.import_module(neural_network)

filenames = []

if(args.img):
  print('image_name', args.img)
  fobj = {}
  fobj["img"] = args.img
  fobj["out"] = args.out
  filenames.append(fobj)
elif(args.dir):
  print('dir', args.dir)
  for img in glob.iglob(os.path.join(args.dir, '**/*'), recursive=True):
    fobj = {}
    fobj["img"] = img
    fobj["out"] = os.path.join(args.out, os.path.basename(img))
    filenames.append(fobj)


InputType = itk.Image[itk.F,2]
img_read = itk.ImageFileReader[InputType].New(FileName=filenames[0]["img"])
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

  if is_gan:
    with tf.variable_scope("generator"):
      y_conv = nn.inference(x, keep_prob=1.0, is_training=False, ps_device=ps_device, w_device=w_device)
  else:
    y_conv = nn.inference(x, keep_prob=1.0, is_training=False, ps_device=ps_device, w_device=w_device)

  label = y_conv

  with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    # specify where to write the log files for import to TensorBoard
    now = datetime.now()

    print("I am self aware!")

    for img_obj in filenames:

      print("Predicting:", img_obj["img"])

      img_read = itk.ImageFileReader[InputType].New(FileName=img_obj["img"])
      img_read.Update()
      img = img_read.GetOutput()

      img_np = itk.GetArrayViewFromImage(img)

      label_map = sess.run([label], feed_dict={x: np.reshape(img_np, img_shape)})

      label_map = np.reshape(label_map[0], img_np.shape)
      label_map = np.absolute(label_map)

      img_np.setfield(label_map,img_np.dtype)

      print("Writing:", img_obj["out"])
      writer = itk.ImageFileWriter.New(FileName=img_obj["out"], Input=img)
      writer.Update()

    print("jk, bye")
        