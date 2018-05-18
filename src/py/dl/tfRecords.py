import itk
import argparse
import os
import glob
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import json

def _int64_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(args):
	
	filenames = []

	if(args.img):

		for img in glob.iglob(os.path.join(args.img, '**/*.nrrd'), recursive=True):
			fobj = {}
			fobj["img"] = img

			labeldir = os.path.dirname(img)

			if(args.label):
				labeldir = args.label

			fobj["label"] = os.path.join(labeldir, os.path.splitext(args.prefix + os.path.basename(img))[0] + args.sufix + ".nrrd")
			filenames.append(fobj)

	img_shape = None

	InputType = itk.Image[itk.SS,2]

	if not os.path.exists(args.out):
		os.makedirs(args.out)

	for fobj in filenames:
		img_read = itk.ImageFileReader[InputType].New(FileName=fobj["img"])
		img_read.Update()
		img = img_read.GetOutput()
		
		img_np = itk.GetArrayViewFromImage(img)

		image = itk.GetImageFromArray(img_np)

		if(img_shape is None):
			img_shape = img_np.shape

		label_read = itk.ImageFileReader[InputType].New(FileName=fobj["label"])
		label_read.Update()
		label = label_read.GetOutput()

		label_np = itk.GetArrayViewFromImage(label)
		
		print("Writing record", fobj)

		tfrecords_filename = os.path.join(args.out, os.path.splitext(os.path.basename(fobj["img"]))[0] + ".tfrecords")
		writer = tf.python_io.TFRecordWriter(tfrecords_filename)

		example = tf.train.Example(features=tf.train.Features(feature={
	        'image': _bytes_feature(img_np.tostring()),
	        'label': _bytes_feature(label_np.tostring())
	        }))

		writer.write(example.SerializeToString())

		writer.close()

	obj = {}
	obj['shape'] = list(img_shape)
	obj['tfrecords'] = tfrecords_filename

	with open(args.out, "w") as f:
		f.write(json.dumps(obj))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	in_group = parser.add_mutually_exclusive_group(required=True)
	
	in_group.add_argument('--img', type=str, help='Directory with nrrd images, image has <imagename>.nrrd. All images must have the same dimensions!.')
	in_group.add_argument('--csv', type=str, help='CSV file with two columns: img,label')

	parser.add_argument('--label', type=str, help='Directory with nrrd images, same filename as in "img" directory to match corresponding pairs')
	parser.add_argument('--prefix', type=str, default="", help="Add a prefix to the label filename, seg_ or label_ for example")
	parser.add_argument('--sufix', type=str, default="", help="Add a sufix to the label filename, _seg or _label for example")

	parser.add_argument('--out', type=str, default="./out", help="Output directory for tfrecord files")

	args = parser.parse_args()

	main(args)