import numpy as np
import argparse
import importlib
import os
from datetime import datetime, time
import json
import glob
from tf_inputs_v2 import *
import time
import tensorflow as tf

def main(args):
	# Input Group
	json_filename = args.json

	# Validation Group
	json_filename_validation = args.json_v
	patience_validation = args.patience

	#Output Group
	outvariablesdirname = args.out
	modelname = args.model

	# Train params
	neural_network = args.nn
	neural_network2 = args.nn2
	in_model2_svf = args.in_model2_svf

	# In models to continue training
	in_model = args.in_model
	in_model2 = args.in_model2

	# Train params
	batch_size = args.batch_size
	num_epochs = args.num_epochs
	buffer_size = args.buffer_size

	save_model = args.save_model

	print("Using gpu:", args.gpu)
	with tf.device('/device:GPU:' + str(args.gpu)):

		tf_inputs = TFInputs(json_filename=json_filename, batch_size=batch_size, buffer_size=buffer_size)

		NN = importlib.import_module("nn_v2." + neural_network).NN
		nn = NN(tf_inputs, args)

		use_validate = False
		global_valid_metric = 0
		global_valid_metric_patience = 0

		if json_filename_validation is not None:
			if "valid_step" not in dir(NN):
				raise "valid_step function not implemented. It should return true or false if it improves the evaluation"
			use_validate = True
			tf_inputs_v = TFInputs(json_filename=json_filename_validation, batch_size=batch_size, buffer_size=0)

		if(neural_network2):
			NN2 = importlib.import_module("nn_v2." + neural_network2).NN
			nn2 = NN2(tf_inputs, args)
			if(in_model2):
				print("loading nn2:", in_model2)
				latest = tf.train.latest_checkpoint(in_model2)
				nn2.load_weights(latest)

			nn.set_nn2(nn2)

		if(in_model2_svf):
			nn2 = tf.keras.models.load_model(in_model2_svf)
			nn2.summary()
			nn.set_nn2(nn2)
			

		# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		# model = nn.get_symbolic_model()
		# model.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

		# now = datetime.now()

		# checkpoint_path = os.path.join(outvariablesdirname, modelname)
		# summary_path = os.path.join(outvariablesdirname, modelname   + "-" + now.strftime("%Y%m%d-%H%M%S"))

		# # history = model.fit(dataset, epochs=num_epochs, callbacks=nn.callbacks(checkpoint_path, summary_path))

		if(in_model):
			latest = tf.train.latest_checkpoint(in_model)
			print("loading:", latest)
			nn.load_weights(latest)

		if(save_model):
			print("Saving model to:", save_model)
			nn.save_model(save_model)
			with open(args.json, "r") as r:
				with open(os.path.join(save_model, "data_description.json"), "w") as w:
					w.write(json.dumps(json.load(r)))
		else:

			# Search/get the tfRecords from the directory
			tf_inputs.get_tf_records()

			summary_writer = tf.summary.create_file_writer(os.path.join(outvariablesdirname, modelname + "-" + str(datetime.now())))
			checkpoint_path = os.path.join(outvariablesdirname, modelname)

			ckpt = nn.get_checkpoint_manager()

			checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

			if use_validate:
				checkpoint_path_valid = os.path.join(outvariablesdirname, modelname + "_valid")
				ckpt_valid = nn.get_checkpoint_manager()
				checkpoint_manager_valid = tf.train.CheckpointManager(ckpt_valid, checkpoint_path_valid, max_to_keep=3, checkpoint_name=modelname + "_min")

				tf_inputs_v.get_tf_records()
				dataset_validation = tf_inputs_v.tf_inputs()

			with summary_writer.as_default():

				step = args.in_step
				for epoch in range(args.in_epoch, num_epochs):
					start = time.time()

					dataset = tf_inputs.tf_inputs()

					for image_batch in dataset:

						tr_strep = nn.train_step(image_batch)
						step+=1

						if step % args.summary_writer == 0:
							nn.summary(image_batch, tr_strep, step)
							summary_writer.flush()

					print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

					if use_validate:
						print("Start evaluation")
						if nn.valid_step(dataset_validation):
							global_valid_metric_patience = 0
							ckpt_save_path_valid = checkpoint_manager_valid.save()
							print("Validation metric improved!")
							print("Saving checkpoint validation", ckpt_save_path_valid)					
						else:
							global_valid_metric_patience += 1
							print("Validation did not improve:", global_valid_metric_patience)

						if global_valid_metric_patience >= patience_validation:
							break
					else: 
						ckpt_save_path = checkpoint_manager.save()
						print("Saving checkpoint", ckpt_save_path)

			ckpt_save_path = checkpoint_manager.save()
			print("Saving last checkpoint", ckpt_save_path)


if __name__ == "__main__":

	print("Tensorflow version:", tf.__version__)

	parser = argparse.ArgumentParser(description='Train a neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	in_group = parser.add_mutually_exclusive_group(required=True)
	  
	in_group.add_argument('--args', help='JSON file with arguments.', type=str)
	in_group.add_argument('--json', type=str, help='json file with the description of the inputs, generate it with tfRecords.py')

	valid_param_group = parser.add_argument_group('Validation parameters')
	valid_param_group.add_argument('--json_v', type=str, help='json file with the description of the inputs/dataset for validation, generate it with tfRecords.py. The validation is done after every epoch', default=None)
	valid_param_group.add_argument('--patience', type=int, help='If the validation dataset does not improve after this many epochs, the training stops.', default=1)

	output_param_group = parser.add_argument_group('Output')
	output_param_group.add_argument('--out', help='Output dirname for the model', default="./out")
	output_param_group.add_argument('--model', help='Output modelname, the output name will be <out directory>/model-<num step>', default="model")

	train_param_group = parser.add_argument_group('Training parameters')
	train_param_group.add_argument('--nn', type=str, help='Type of neural network to use', default=None)
	train_param_group.add_argument('--nn2', type=str, help='Type of neural network to use', default=None)

	train_param_group.add_argument('--drop_prob', help='The probability that each element is dropped during training', type=float, default=0.0)
	train_param_group.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-03)
	train_param_group.add_argument('--decay_rate', help='decay rate', type=float, default=0.0)
	train_param_group.add_argument('--decay_steps', help='decay steps', type=int, default=1000)
	train_param_group.add_argument('--staircase', help='staircase decay', type=bool, default=False)
	train_param_group.add_argument('--batch_size', help='Batch size for evaluation', type=int, default=8)
	train_param_group.add_argument('--num_epochs', help='Number of epochs', type=int, default=10)
	train_param_group.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=0)
	train_param_group.add_argument('--summary_writer', help='Number of steps to write summary', type=int, default=100)
	train_param_group.add_argument('--gpu', help='GPU number for training', type=int, default=0)
	train_param_group.add_argument('--train_cross', type=bool, help='Train multiple models using multiple folds. The folds are created using the tfRecords.py. Use the input json without the "fold_n" name as input to the json parameter.', default=None)

	continue_param_group = parser.add_argument_group('Continue training', 'Use a previously saved model to continue the training.')
	continue_param_group.add_argument('--in_model', help='Input model name', default=None)
	continue_param_group.add_argument('--in_step', help='Input step', type=int, default=0)
	continue_param_group.add_argument('--in_epoch', help='Input epoch', type=int, default=0)
	continue_param_group.add_argument('--in_model2', help='Input model name', default=None)
	continue_param_group.add_argument('--in_model2_svf', help='Input model 2 but in save model format', default=None)

	export_param_group = parser.add_argument_group('Export as save_model format')
	export_param_group.add_argument('--save_model', help='Export folder', default=None)

	args = parser.parse_args()


	if args.train_cross:
		json_folds = os.path.splitext(args.json)[0] + '*fold*.json'
		out_dir = args.out
		save_model = args.save_model
		for jsf in glob.iglob(json_folds):
			args.json = jsf
			out_name = os.path.splitext(os.path.basename(jsf))[0]
			args.out = os.path.join(out_dir, out_name)
			if args.save_model:
				args.in_model = os.path.join(args.out, args.model)
				args.save_model = os.path.join(save_model, out_name)
			main(args)
	else:
		main(args)
