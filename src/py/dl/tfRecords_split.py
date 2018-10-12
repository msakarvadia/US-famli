import argparse
import os
import glob
import numpy as np
import sys
import json

def main(args):
	
	json_filename = args.json
	data_description = {}

	with open(json_filename, "r") as f:
		data_description = json.load(f)

	tfrecords_arr = []
	tfrecords_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"], '**/*.tfrecord')
	for tfr in glob.iglob(tfrecords_dir, recursive=True):
		tfrecords_arr.append(tfr)

	tfrecords_arr = np.array(tfrecords_arr)
	tfrecords_arr = tfrecords_arr[np.random.permutation(tfrecords_arr.size)]

	train_samples = int(tfrecords_arr.size - tfrecords_arr.size*args.split)

	tfrecords_train_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"] + "_split_train")
	data_description["tfrecords"] = os.path.basename(tfrecords_train_dir)

	if(not os.path.exists(tfrecords_train_dir)):
		os.makedirs(tfrecords_train_dir)

	for tfr in tfrecords_arr[0:train_samples]:
		try:
			dest = os.path.join(tfrecords_train_dir, os.path.basename(tfr))
			print("Creating soft link: ", dest)
			os.symlink(tfr, dest)
		except Exception as e:
			print("Error making symlink", e, file=sys.stderr)

	tfrecords_train_json = tfrecords_train_dir.rstrip(os.sep) + ".json"
	with open(tfrecords_train_json, "w") as f:
		f.write(json.dumps(data_description))

	tfrecords_eval_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"] + "_eval")
	data_description["tfrecords"] = os.path.basename(tfrecords_eval_dir)

	if(not os.path.exists(tfrecords_eval_dir)):
		os.makedirs(tfrecords_eval_dir)

	for tfr in tfrecords_arr[train_samples:]:
		try:
			dest = os.path.join(tfrecords_eval_dir, os.path.basename(tfr))
			print("Creating soft link: ", dest)
			os.symlink(tfr, dest)
		except Exception as e:
			print("Error making symlink", e, file=sys.stderr)
	
	tfrecords_eval_json = tfrecords_eval_dir.rstrip(os.sep) + ".json"
	with open(tfrecords_eval_json, "w") as f:
		f.write(json.dumps(data_description))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Splits the data by creating soft links and new folders for training and evaluation purposes')
	
	parser.add_argument('--json', type=str, help='JSON file created by tfRecords.py', required=True)
	parser.add_argument('--split', type=float, default=0, help="Split the data for evaluation. [0-1], 0=no split")

	args = parser.parse_args()

	main(args)
