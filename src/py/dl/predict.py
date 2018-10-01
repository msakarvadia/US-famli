
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
import csv

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='U net segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

in_group = parser.add_mutually_exclusive_group(required=True)
in_group.add_argument('--img', type=str, help='Input image for prediction')
in_group.add_argument('--dir', type=str, help='Directory with images for prediction')

parser.add_argument('--json', help='JSON file with model description, created by train.py', required=True)

parser.add_argument('--out', type=str, help='Output image, csv, or directory. If --dir flag is used the output image name will be the <Directory set in out flag>/<imgage filename in directory dir>', default="out")
parser.add_argument('--out_ext', type=str, help='Output extension for images', default='.nrrd')
parser.add_argument('--ow', type=int, help='Overwrite outputs', default=1)
parser.add_argument('--resize', nargs="+", type=int, help='Resize images during prediction, useful when doing whole directories with images of diferent sizes. This is needed to set the value of the placeholder for tensorflow. e.x. 1500 1500. The image will be resized for the prediction but the original image size will be stored. Do not include the channels/pixel components in the resize parameters', default=None)
parser.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
parser.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

args = parser.parse_args()

json_model_name = args.json
out_name = args.out
out_ext = args.out_ext
resize_shape = args.resize
ps_device = args.ps_device
w_device = args.w_device

with open(json_model_name, "r") as f:
  model_description = json.load(f)
  model_name = os.path.join(os.path.dirname(json_model_name), model_description["model"])
  neural_network = model_description["nn"]

print('json', json_model_name)
print('model_name', model_name)
print('neural_network', neural_network)
is_gan = "gan" in neural_network
if is_gan:
  print('Using gan scheme')
print('ps_device', ps_device)
print('w_device', w_device)
print('ow', args.ow)
print('out', out_name)
print('out_ext', out_ext)

nn = importlib.import_module("nn." + neural_network).NN()

if("description" in model_description):
  nn.set_data_description(data_description=model_description["description"])

class_prediction = False
class_prediction_arr = []
if("description" in model_description and "enumerate" in model_description["description"]):
  class_prediction = True
  class_obj = {}
  enumerate_obj = model_description["description"][model_description["description"]["enumerate"]]["class"]
  for key in enumerate_obj:
    class_obj[enumerate_obj[key]] = key 

prediction_arr = []

filenames = []

if(args.img):
  print('image_name', args.img)
  fobj = {}
  fobj["img"] = args.img
  fobj["out"] = out_name
  if args.ow or not os.path.exists(fobj["out"]):
    filenames.append(fobj)
elif(args.dir):
  
  normpath = os.path.normpath("/".join([args.dir, '**', '*']))

  for img in glob.iglob(normpath, recursive=True):
    if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".jpg", ".png"]]:
      fobj = {}
      fobj["img"] = img

      if(not class_prediction):
        image_dir_filename = img.replace(args.dir, '')
        image_dir_filename = os.path.splitext(image_dir_filename)[0] +  out_ext
        fobj["out"] = os.path.normpath("/".join([out_name, image_dir_filename]))

        if not os.path.exists(os.path.dirname(fobj["out"])):
          os.makedirs(os.path.dirname(fobj["out"]))

      if args.ow or not os.path.exists(fobj["out"]):
        filenames.append(fobj)

if len(filenames) == 0:
  print("No images found with extensions", [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".jpg", ".png"], file=sys.stderr)
  sys.exit()

def image_read(filename):
  img_read = itk.ImageFileReader.New(FileName=filename)
  img_read.Update()
  img = img_read.GetOutput()
  
  img_np = itk.GetArrayViewFromImage(img).astype(float)

  # Put the shape of the image in the json object if it does not exists. This is done for global information
  tf_img_shape = list(img_np.shape)
  if(tf_img_shape[0] == 1):
    # If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
    tf_img_shape = tf_img_shape[1:]

  # This is the number of channels, if the number of components is 1, it is not included in the image shape
  # If it has more than one component, it is included in the shape, that's why we have to add the 1
  if(img.GetNumberOfComponentsPerPixel() == 1):
    tf_img_shape = tf_img_shape + [1]

  tf_img_shape = [1] + tf_img_shape

  return img, img_np, tf_img_shape

try:
  _itkimg, _imgnp, tf_img_shape = image_read(filenames[0]["img"])
except Exception as e:
  print("Error reading image to get shape info for prediction", filenames[0]["img"], e, file=sys.stderr)
  sys.exit()

if(resize_shape):
  resize_shape = tuple(resize_shape)
  tf_img_shape = (1,) + resize_shape + (tf_img_shape[-1],)

graph = tf.Graph()

with graph.as_default():

  x = tf.placeholder(tf.float32,shape=tf_img_shape)
  
  keep_prob = tf.placeholder(tf.float32)

  if is_gan:
    with tf.variable_scope("generator"):
      y_conv = nn.inference(images=x, keep_prob=1.0, is_training=False, ps_device=ps_device, w_device=w_device)
  else:
    y_conv = nn.inference(images=x, keep_prob=1.0, is_training=False, ps_device=ps_device, w_device=w_device)

  label = nn.predict(y_conv)

  with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    print("I am self aware!")

    for img_obj in filenames:

      print("Predicting:", img_obj["img"])

      try:

        img, img_np, img_shape_current = image_read(img_obj["img"])

        if(resize_shape):
          
          img_np_x = np.zeros(tf_img_shape)

          prediction_shape = []
          for r_s, i_s in zip(tf_img_shape, img_shape_current):
            if(r_s >= i_s):
              prediction_shape.append("0:" + str(i_s))
            else:
              print("The resize shape is smaller than the current image shape...", img_obj["img"], file=sys.stderr)
              prediction_shape.append("0:")

          assign_img = "img_np_x[" + ",".join(prediction_shape) + "]=np.reshape(img_np, img_shape_current)"
          exec(assign_img)
        else:
          img_np_x = np.reshape(img_np, tf_img_shape)

        label_map = sess.run([label], feed_dict={x: img_np_x.astype(np.float32)})
        label_map = np.array(label_map)[0]

        if(class_prediction):
          class_prediction_arr.append({
            "img": img_obj["img"],
            "class": class_obj[label_map[0][0]],
            "prob": label_map[1][0]
            })
        elif(nn.prediction_type() == "image"):
          if(resize_shape):
            assign_img = "label_map=label_map[" + ",".join(prediction_shape) + "]"
            exec(assign_img)

          #THE NUMBER OF CHANNELS OF THE OUTPUT ARE GIVEN BY THE NEURAL NET
          label_map_reshape = list(img_np.shape)
          if(img.GetNumberOfComponentsPerPixel() > 1):
            label_map_reshape[-1] = np.array(label_map).shape[-1]
          
          
          label_map = np.reshape(label_map, label_map_reshape)
          label_map = np.absolute(label_map)
          label_map = np.around(label_map).astype(np.uint16)

          Dimension = len(img.GetLargestPossibleRegion().GetSize())
          PixelType = itk.ctype('unsigned short')
          OutputImageType = itk.Image[PixelType, Dimension]

          out_img = OutputImageType.New()

          out_img.SetRegions(img.GetLargestPossibleRegion())
          out_img.SetDirection(img.GetDirection())
          out_img.SetOrigin(img.GetOrigin())
          out_img.SetSpacing(img.GetSpacing())
          out_img.Allocate()

          out_img_np = itk.GetArrayViewFromImage(out_img)

          out_img_np.setfield(np.reshape(label_map, out_img_np.shape), out_img_np.dtype)

          print("Writing:", img_obj["out"])
          writer = itk.ImageFileWriter.New(FileName=img_obj["out"], Input=out_img)
          writer.Update()

        elif(nn.prediction_type() == "scalar"):
          prediction_arr.append({
            "img": img_obj["img"],
            "scalar": label_map[0].tolist()
            })
        else:
          print("I don't know what to do with the result, 'ill just printed")

      except Exception as e:
        print("Error predicting image:", e, file=sys.stderr)
        print("Continuing...", file=sys.stderr)

    if(class_prediction):
      with open(out_name, 'w') as csvfile:
        fieldnames = ['img', 'class', 'prob']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for cp in class_prediction_arr:
          writer.writerow(cp)

    if(prediction_arr):
      print(prediction_arr)

    print("jk, bye")
        