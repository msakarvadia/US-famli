
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import importlib
import os
from datetime import datetime, time
import json
import glob
from tf_inputs_v2 import *
import time

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='U network for segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

in_group = parser.add_mutually_exclusive_group(required=True)
  
in_group.add_argument('--args', help='JSON file with arguments.', type=str)
in_group.add_argument('--json', type=str, help='json file with the description of the inputs, generate it with tfRecords.py')

output_param_group = parser.add_argument_group('Output')
output_param_group.add_argument('--out', help='Output dirname for the model', default="./out")
output_param_group.add_argument('--model', help='Output modelname, the output name will be <out directory>/model-<num step>', default="model")

train_param_group = parser.add_argument_group('Training parameters')
train_param_group.add_argument('--nn', type=str, help='Type of neural network to use', default=None)
train_param_group.add_argument('--nn2', type=str, help='Type of neural network to use', default=None)
train_param_group.add_argument('--drop_prob', help='The probability that each element is dropped during training', type=float, default=0.0)
train_param_group.add_argument('--learning_rate', help='Learning rate, default=1e-5', type=float, default=1e-03)
train_param_group.add_argument('--learning_rate_discriminator_mul', help='Factor for the learning rate for the discriminator (to make it slower/faster)', type=float, default=1)
train_param_group.add_argument('--skip_steps', help='Number of skip steps for gan_512 training', type=int, default=10)
train_param_group.add_argument('--decay_rate', help='decay rate, default=0.96', type=float, default=0.96)
train_param_group.add_argument('--decay_steps', help='decay steps, default=10000', type=int, default=1000)
train_param_group.add_argument('--staircase', help='staircase decay', type=bool, default=False)
train_param_group.add_argument('--batch_size', help='Batch size for evaluation', type=int, default=8)
train_param_group.add_argument('--num_epochs', help='Number of epochs', type=int, default=10)
train_param_group.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=1000)
train_param_group.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
train_param_group.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

continue_param_group = parser.add_argument_group('Continue training', 'Use a previously saved model to continue the training.')
continue_param_group.add_argument('--restore_all', help='Restore all variables from the model or partially (check the implementation of restore for the given nn)', type=bool, default=False)
continue_param_group.add_argument('--in_model', help='Input model name', default=None)
continue_param_group.add_argument('--in_step', type=int, help='Set the step number to start', default=0)
continue_param_group.add_argument('--in_model2', help='Input model name', default=None)

export_param_group = parser.add_argument_group('Export as save_model format')
export_param_group.add_argument('--save_model', help='Export folder', default=None)

args = parser.parse_args()

json_filename = args.json
neural_network = args.nn
neural_network2 = args.nn2
outvariablesdirname = args.out
modelname = args.model
drop_prob = args.drop_prob
learning_rate = args.learning_rate
learning_rate_discriminator_mul = args.learning_rate_discriminator_mul
decay_rate = args.decay_rate
decay_steps = args.decay_steps
staircase = args.staircase
batch_size = args.batch_size
num_epochs = args.num_epochs
buffer_size = args.buffer_size
skip_steps = args.skip_steps
ps_device = args.ps_device
w_device = args.w_device

in_model = args.in_model
in_model2 = args.in_model2
in_step = args.in_step
restore_all = args.restore_all

save_model = args.save_model

tf_inputs = TFInputs(json_filename=json_filename)
dataset = tf_inputs.tf_inputs(batch_size=batch_size, buffer_size=buffer_size)

NN = importlib.import_module("nn_v2." + neural_network).NN
nn = NN(tf_inputs, learning_rate = learning_rate, decay_steps = decay_steps, decay_rate = decay_rate, staircase = staircase, drop_prob = drop_prob)

if(neural_network2):
	NN2 = importlib.import_module("nn_v2." + neural_network2).NN
	nn2 = NN2(tf_inputs, learning_rate = learning_rate, decay_steps = decay_steps, decay_rate = decay_rate, staircase = staircase, drop_prob = drop_prob)
	if(in_model2):
		latest = tf.train.latest_checkpoint(in_model2)
		nn2.load_weights(latest)

	nn.set_nn2(nn2)

# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# model = nn.get_symbolic_model()
# model.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# now = datetime.now()

# checkpoint_path = os.path.join(outvariablesdirname, modelname)
# summary_path = os.path.join(outvariablesdirname, modelname   + "-" + now.strftime("%Y%m%d-%H%M%S"))

# # history = model.fit(dataset, epochs=num_epochs, callbacks=nn.callbacks(checkpoint_path, summary_path))

summary_writer = tf.summary.create_file_writer(os.path.join(outvariablesdirname, modelname + "-" + str(datetime.now())))
checkpoint_path = os.path.join(outvariablesdirname, modelname)

ckpt = nn.get_checkpoint_manager()

checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

if(in_model):
	latest = tf.train.latest_checkpoint(in_model)
	nn.load_weights(latest)

if(save_model):
	nn.save_model(save_model)
else:
	with summary_writer.as_default():

		step = 0
		for epoch in range(num_epochs):
			start = time.time()

			for image_batch in dataset:

				tr_strep = nn.train_step(image_batch)
				step+=1

				if step % 100 == 0:
					nn.summary(image_batch, tr_strep, step)
					summary_writer.flush()

				if step % 1000 == 0:
					ckpt_save_path = checkpoint_manager.save()
					print("Saving checkpoint", ckpt_save_path)

			print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


	ckpt_save_path = checkpoint_manager.save()
	print("Saving last checkpoint", ckpt_save_path)
