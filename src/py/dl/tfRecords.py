import itk
import argparse
import os
import glob
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import sys
import json
import csv
import uuid

def _int64_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def main(args):
	

	print("Input csv", args.csv)
	if(args.enumerate):
		print("Enumerate", args.enumerate)
	print("Output", args.out)

	csv_rows = []

	obj = {}

	with open(args.csv) as csvfile:

		csv_reader = csv.DictReader(csvfile)
		for row in csv_reader:
			csv_rows.append(row)

		row_keys = csv_reader.fieldnames.copy()

		if("tfRecord" in row_keys):
			row_keys.remove("tfRecord")
		if(not "data_keys" in obj):
			obj["data_keys"] = row_keys

	if(not os.path.exists(args.out) or not os.path.isdir(args.out)):
		os.makedirs(args.out)

	if(args.enumerate):
		obj[args.enumerate] = {}
		class_number = 0
		for fobj in csv_rows:
			class_name = fobj[args.enumerate]
			if(not class_name in obj[args.enumerate]):
				obj[args.enumerate][class_name] = class_number
				class_number += 1

		print(obj[args.enumerate])

	for fobj in csv_rows:
		
		feature = {}

		try:

			for key in row_keys:

				##If the path exists then it will try to read it as an image
				if(os.path.exists(fobj[key])):

					if(not key in obj):
						obj[key] = {}

					img_read = itk.ImageFileReader.New(FileName=fobj[key])
					img_read.Update()
					img = img_read.GetOutput()
					
					img_np = itk.GetArrayViewFromImage(img).astype(float)
					feature[key] =  _float_feature(img_np.reshape(-1).tolist())

					# Put the shape of the image in the json object if it does not exists. This is done for global information
					img_shape = list(img_np.shape)
					if(img_shape[0] == 1):
						# If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
						img_shape = img_shape[1:]

					# This is the number of channels, if the components is 1, is not included in the image shape
					# If it has more than one component, it is included in the shape
					if(img.GetNumberOfComponentsPerPixel() == 1):
						img_shape = img_shape + [1]

					if(not "shape" in obj[key]):
						obj[key]["shape"] = img_shape
					else:
						obj[key]["shape"] = np.maximum(obj[key]["shape"], img_shape).tolist()

					if(not "max" in obj[key]):
						obj[key]["max"] = np.max(img_np)
					else:
						obj[key]["max"] = max(obj[key]["max"], np.max(img_np))

					if(not "min" in obj[key]):
						obj[key]["min"] = np.min(img_np)
					else:
						obj[key]["min"] = min(obj[key]["min"], np.min(img_np))

					if(not "type" in obj[key]):
						obj[key]["type"] = "tf.float32"
					
				elif(args.enumerate and key == args.enumerate):
					class_name = fobj[key]
					class_number = int(obj[args.enumerate][class_name])

					feature[key] = _int64_feature(class_number)

					if(not "shape" in obj[key]):
						obj[key]["shape"] = [1]

					if(not "type" in obj[key]):
						obj[key]["type"] = "tf.int32"

				else:
					#If is not an image, try to put what ever data is in there into the TF Record
					try:
						# Try converting the data to int, if it is a float it will fail
						feature[key] = _int64_feature(int(fobj[key]))

						if(not "shape" in obj[key]):
							obj[key]["shape"] = [1]

						if(not "type" in obj[key]):
							obj[key]["type"] = "tf.int32"
					except:
						try:
							# If the previous failed, try converting it to float
							feature[key] = _float_feature(float(fobj[key]))

							if(not "shape" in obj[key]):
								obj[key]["shape"] = [1]

							if(not "type" in obj[key]):
								obj[key]["type"] = "tf.float32"
						except:
							# If it fails the try saving it as a bytes feature
							# Convert the string to a list
							str_list = list(fobj[key])
							feature[key] = _bytes_feature(str_list)

							if(not "shape" in obj[key]):
								obj[key]["shape"] = [len(str_list)]

							if(not "type" in obj[key]):
								obj[key]["type"] = "tf.string"
		
			if("tfRecord" in fobj):
				record_path = fobj["tfRecord"]
			else:
				record_path = os.path.join(args.out, str(uuid.uuid4()) + ".tfrecord")

			fobj["tfRecord"] = record_path

			writer = tf.python_io.TFRecordWriter(record_path)
			example = tf.train.Example(features=tf.train.Features(feature=feature))

			print("Writing record", fobj)

			writer.write(example.SerializeToString())
			writer.close()

		except Exception as e:
			print("Error converting to tfRecord", obj, e, file=sys.stderr)
			print("I'll keep on going...")

	obj['tfrecords'] = os.path.basename(args.out.rstrip(os.sep))
	
	outjson = args.out.rstrip(os.sep) + ".json"
	print("Writing:", outjson)

	print(obj)
	with open(outjson, "w") as f:
		f.write(json.dumps(obj))
	
	if(len(csv_rows) > 0):
		outcsv = os.path.splitext(args.csv)[0] + "_tfRecords.csv"
		print("Writing:", outcsv)
		csv_headers = list(csv_rows[0].keys())
		print(csv_headers)
		with open(outcsv, "w") as f:
			writer = csv.DictWriter(f, fieldnames=csv_headers)
			writer.writeheader()
			for row in csv_rows:
				writer.writerow(row)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Writes data to tfrecords format using a CSV file description as input and creates a JSON file with a description. The column headers are used as keys to store in the tfRecord. a row may contain image filenames to save in tfRecords format. If an image column is found, the maximum pixel value for each image is calulated and written into the json file. Additionally, it will save a new csv file indicating which tfRecord corresponds to the row.')
	
	parser.add_argument('--csv', type=str, help='CSV file with dataset information,', required=True)
	parser.add_argument('--enumerate', type=str, help='Column name in CSV. Enumerate the elements in the column, i.e., in case you are using text labels to identify the classes of your data, an enumeration will be made before storing into tfRecords.')
	parser.add_argument('--out', type=str, default="./out", help="Output directory")

	args = parser.parse_args()

	main(args)
