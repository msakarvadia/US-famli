
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
import sys

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='U net segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


in_group = parser.add_mutually_exclusive_group(required=True)
  
in_group.add_argument('--img', type=str, help='Input image for prediction')
in_group.add_argument('--dir', type=str, help='Directory with images for prediction')

parser.add_argument('--out', type=str, help='Output image or directory. If dir flag is used the output image name will be the <Directory set in out flag>/<imgage filename in directory dir>', default="./out.nrrd")
parser.add_argument('--ow', type=int, help='Overwrite outputs', default=1)
parser.add_argument('--resize', nargs="+", type=int, help='Resize images during prediction, useful when doing whole directories with images of diferent sizes. This is needed to set the value of the placeholder for tensorflow. e.x. 1500 1500. The image will be resized for the prediction but the original image size will be stored. Do not include the channels/pixel components in the resize parameters', default=None)
parser.add_argument('--model', help='Input modelname', required=True)
parser.add_argument('--nn', type=str, help='Type of neural network to use', default='u_nn')
parser.add_argument('--num_labels', help='Number of labels for the softmax output', type=int, default=2)
parser.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
parser.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

args = parser.parse_args()


model_name = args.model
neural_network = args.nn
out_name = args.out
resize_shape = args.resize
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
print('ow', args.ow)
print('out', out_name)

nn = importlib.import_module(neural_network)

filenames = []

if(args.img):
  print('imgage_name', args.img)
  fobj = {}
  fobj["img"] = args.img
  fobj["out"] = args.out
  if args.ow or not os.path.exists(fobj["out"]):
    filenames.append(fobj)
elif(args.dir):
  
  normpath = os.path.normpath("/".join([args.dir, '**', '*']))

  for img in glob.iglob(normpath, recursive=True):
    if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".jpg", ".png"]]:
      fobj = {}
      fobj["img"] = img

      image_dir_filename = img.replace(args.dir, '')
      fobj["out"] = os.path.normpath("/".join([args.out, image_dir_filename]))

      if not os.path.exists(os.path.dirname(fobj["out"])):
        os.makedirs(os.path.dirname(fobj["out"]))

      if args.ow or not os.path.exists(fobj["out"]):
        filenames.append(fobj)

if len(filenames) == 0:
  print("No images found with extensions", [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".jpg", ".png"], file=sys.stderr)
  sys.exit()

components_pixel = 1
try:
  InputType = itk.Image[itk.US,2]
  img_read = itk.ImageFileReader[InputType].New(FileName=filenames[0]["img"])
  img_read.Update()
  img = img_read.GetOutput()
  components_pixel = img.GetNumberOfComponentsPerPixel()
except Exception as e:
  print("Error predicting image", filenames[0].img, e, file=sys.stderr)
  sys.exit()

if(resize_shape is None):
  img_np = itk.GetArrayViewFromImage(img)
  if(components_pixel == 1):
    img_shape = (1,) + img_np.shape + (1,)
  else:
    img_shape = (1,) + img_np.shape 
else:
  resize_shape = tuple(resize_shape)
  img_shape = (1,) + resize_shape + (components_pixel,)

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

    print("I am self aware!")

    for img_obj in filenames:

      print("Predicting:", img_obj["img"])

      try:
        img_read = itk.ImageFileReader[InputType].New(FileName=img_obj["img"])
        img_read.Update()
        img = img_read.GetOutput()

        img_np = itk.GetArrayViewFromImage(img)
        img_shape_current = img_np.shape

        if(resize_shape):
          
          img_np_x = np.zeros(img_shape)

          if(img.GetNumberOfComponentsPerPixel() == 1):
            img_reshape_current = (1,) + img_shape_current + (1,)
          else:
            img_reshape_current = (1,) + img_shape_current

          prediction_shape = []
          for r_s, i_s in zip(img_shape, img_reshape_current):
            if(r_s >= i_s):
              prediction_shape.append("0:" + str(i_s))
            else:
              print("The resize shape is smaller than the current image shape...", img_obj["img"], file=sys.stderr)
              prediction_shape.append("0:")

          assign_img = "img_np_x[" + ",".join(prediction_shape) + "]=np.reshape(img_np, img_reshape_current)"
          exec(assign_img)

        else:
          img_np_x = np.reshape(img_np, img_shape)

        label_map = sess.run([label], feed_dict={x: img_np_x.astype(np.float32)})

        if(resize_shape):
          label_map = np.reshape(label_map, img_shape)
          assign_img = "label_map=label_map[" + ",".join(prediction_shape) + "]"
          print(assign_img)
          exec(assign_img)

        label_map = np.reshape(label_map, img_shape_current)
        label_map = np.absolute(label_map)
        label_map = label_map.astype(np.uint16)

        img_np.setfield(label_map,img_np.dtype)

        print("Writing:", img_obj["out"])
        writer = itk.ImageFileWriter.New(FileName=img_obj["out"], Input=img)
        writer.Update()
      except Exception as e:
        print("Error predicting image", e)
        print("Continuing...")

    print("jk, bye")
        