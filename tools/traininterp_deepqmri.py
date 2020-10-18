# Author: Francesco Grussu, University College London
#		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>
#
# Code released under BSD Two-Clause license
#
# Copyright (c) 2020 University College London. 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

### Load libraries
import argparse, os, sys
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch import autograd
import pickle as pk
from pathlib import Path as pt
sys.path.insert(0, os.path.dirname(pt(__file__).absolute()) )
import deepqmri


if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program trains a qMRI-net for quantitative MRI protocol resampling. A qMRI-Nnet enables voxel-by-voxel resampling of a protocol acquired with M measurements to a new protocol acquired with N measurements, with N typically s.t. N > M, or with N = M but coming from a different scanner. Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')
	parser.add_argument('sigin_train', help='path to a pickle binary file storing the training MRI signals acquired with the first protocol as a numpy matrix (rows: voxels; columns: M measurements)')
	parser.add_argument('sigin_val', help='path to a pickle binary file storing the validation MRI signals acquired with the first protocol as a numpy matrix (rows: voxels; columns: M measurements)')
	parser.add_argument('sigout_train', help='path to a pickle binary file storing the training MRI signals acquired with the second protocol (to which input data should be resampled at deployment stage) as a numpy matrix (rows: voxels, with each voxel corresponding 1:1 to voxels in sigin_train; columns: N measurements, typically N > M, or N = M but coming from a different scanner)')
	parser.add_argument('sigout_val', help='path to a pickle binary file storing the validation MRI signals acquired with the second protocol (to which input data should be resampled at deployment stage) as a numpy matrix (rows: voxels, with each voxel corresponding 1:1 to voxels in sigin_val; columns: N measurements, typically N > M, or N = M but coming from a different scanner)')
	parser.add_argument('out_base', help='base name of output directory (a string built with the network parameters will be added to the base). The output directory will contain the following output files: ** losstrain.bin, pickle binary storing the training loss as a numpy matrix (shape: epoch x batch); ** lossval.bin, pickle binary storing the validation loss as a numpy matrix (shape: epoch x 1); ** nnet_epoch0.bin, pickle binary storing the qMRI-net at initialisation; ** nnet_epoch0.pth, Pytorch binary storing the qMRI-net at initialisation; ** nnet_epoch<FINAL_EPOCH>.bin, pickle binary storing the qMRI-net at the final epoch; ** nnet_lossvalmin.bin, pickle binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); * nnet_lossvalmin.pth, Pytorch binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_sigval.bin, prediction of target validation signals sigout_val (shape: voxels x N measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin.info, text file reporting information regarding the epoch with the lowest validation loss; ** nnet_lossvalmin_sigtest.bin, prediction of optional target test signals provided via optional input --dtest (shape: voxels x N measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information), if those signals are provided')
	parser.add_argument('--act', metavar='<value>', default='ReLU', help='non-linear activation function for hidden neurons. Choose among "ReLU", "LeakyReLU", "RReLU", "Sigmoid". Default: "ReLU". More information here http://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)')
	parser.add_argument('--nn', metavar='<list>', help='array storing the number of hidden neurons, separated by hyphens (example: 8-15-30). Default: M-(M + (N minus M)/2)/-N, where M is the number of MRI measurements in the input protocol and N is the number of MRI measurements in the output target protocol')
	parser.add_argument('--pdrop', metavar='<value>', default='0.0', help='dropout probability in each layer of the neural network. Default: 0.0')
	parser.add_argument('--noepoch', metavar='<value>', default='500', help='number of epochs used for training. Default: 500')
	parser.add_argument('--lrate', metavar='<value>', default='0.001', help='learning rate. Default: 0.001')
	parser.add_argument('--mbatch', metavar='<value>', help='number of voxels in each training mini-batch. Default: 1/10 of the total number of training voxels (minimum: 2 voxels)')
	parser.add_argument('--seed', metavar='<value>', default='19102018', help='integer used as a seed for Numpy and PyTorch random number generators. Default: 19102018')
	parser.add_argument('--nwork', metavar='<value>', default='0', help='number of workers for data loader. Default: 0')
	parser.add_argument('--dtest', metavar='<file>', help='path to a pickle binary file storing the test MRI signals acquired with the first protocol as a numpy matrix (rows: voxels; columns: M measurements)')
	args = parser.parse_args()

	### Get some of the inputs
	pdrop = float(args.pdrop)
	noepoch = int(args.noepoch)
	lrate = float(args.lrate)
	seed = int(args.seed)
	nwork = int(args.nwork)
	actfoo = args.act

	### Print some information
	print('')
	print('')
	print('********************************************************************')
	print('                TRAIN A qMRI-NET (qmriinterp CLASS)                 ')
	print('********************************************************************')
	print('')
	print('** Input training MRI signals (M measaurements): {}'.format(args.sigin_train))
	print('** Input validation MRI signals (M measaurements): {}'.format(args.sigin_val))
	print('** Target training MRI signals (N measaurements): {}'.format(args.sigout_train))
	print('** Target validation MRI signals (N measaurements): {}'.format(args.sigout_val))
	if args.dtest is not None:
		print('** Input test MRI signals (N measaurements): {}'.format(args.dtest))

	### Load input training data
	fh = open(args.sigin_train,'rb')
	datatrain = np.float32(pk.load(fh))
	nmeas_in = datatrain.shape[1]
	fh.close()

	### Load input validation data
	fh = open(args.sigin_val,'rb')
	dataval = np.float32(pk.load(fh))
	fh.close()
	if dataval.shape[1]!=datatrain.shape[1]:
		raise RuntimeError('the number of MRI measurements in the input validation set {} differs from the input training set {}!'.format(dataval.shape[1],datatrain.shape[1]))		

	### Load test MRI signals
	if args.dtest is not None:
		fh = open(args.dtest,'rb')
		datatest = np.float32(pk.load(fh))
		fh.close()
		if datatest.shape[1]!=datatrain.shape[1]:
			raise RuntimeError('the number of MRI measurements in the input test set {} differs from the input training set {}!'.format(datatest.shape[1],datatrain.shape[1]))

	### Load target training data
	fh = open(args.sigout_train,'rb')
	datatrain_out = np.float32(pk.load(fh))
	nmeas_out = datatrain_out.shape[1]
	fh.close()

	### Load target validation data
	fh = open(args.sigout_val,'rb')
	dataval_out = np.float32(pk.load(fh))
	fh.close()
	if dataval_out.shape[1]!=datatrain_out.shape[1]:
		raise RuntimeError('the number of MRI measurements in the output (target) validation set {} differs from the output training set {}!'.format(dataval_out.shape[1],datatrain_out.shape[1]))		

	### Get number of mini-batches
	if args.mbatch is None:
		mbatch = int(float(datatrain.shape[0]) / 10.0) # Default: 1/10 of the total number of training voxels
	else:
		mbatch = int(args.mbatch)
		if (mbatch>datatrain.shape[0]):
			mbatch = datatrain.shape[0]
		if(mbatch<2):
			mbatch = int(2)

	### Get specifics for hidden layers
	if args.nn is None:

		nhidden = np.array([int(nmeas_in) , int(float(nmeas_in)+0.5*( float(nmeas_out) - float(nmeas_in))) , int(nmeas_out)])
		nhidden_str = '{}-{}-{}'.format( int(nmeas_in) , int(float(nmeas_in)+0.5*( float(nmeas_out) - float(nmeas_in))) , int(nmeas_out)  )
	else:
		nhidden = (args.nn).split('-')
		nhidden = np.array( list(map( int,nhidden )) )
		nhidden_str = args.nn


	### Create output base name
	out_base_dir = '{}_nhidden{}_pdrop{}_{}_noepoch{}_lr{}_mbatch{}_seed{}'.format(args.out_base,nhidden_str,pdrop,actfoo,noepoch,lrate,mbatch,seed)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir)

	### Print some more information
	print('** Output directory: {}'.format(out_base_dir))
	print('')
	print('')
	print('PARAMETERS')
	print('')
	print('** Non-linear activation: {}'.format(actfoo))
	print('** Hidden neurons: {}'.format(nhidden))
	print('** Dropout probability: {}'.format(pdrop))
	print('** Number of epochs: {}'.format(noepoch))
	print('** Learning rate: {}'.format(lrate))
	print('** Number of voxels in a mini-batch: {}'.format(mbatch))
	print('** Seed: {}'.format(seed))
	print('** Number of workers for data loader: {}'.format(nwork))
	print('')
	print('')


	### Set random seeds
	np.random.seed(seed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(seed)    # Random seed for reproducibility: PyTorch

	### Normalise data
	min_val=0.0
	max_val_train = np.max(datatrain)
	max_val_val = np.max(dataval)
	max_val_train_out = np.max(datatrain_out)
	max_val_val_out = np.max(dataval_out)
	if args.dtest is not None:
		max_val_test = np.max(datatest)
		max_val = np.max(np.array([max_val_train,max_val_val,max_val_train_out,max_val_val_out,max_val_test]))
	else:
		max_val = np.max(np.array([max_val_train,max_val_val,max_val_train_out,max_val_val_out]))

	datatrain[datatrain<min_val] = min_val
	datatrain = np.float32( (datatrain - min_val) / (max_val - min_val) )

	dataval[dataval<min_val] = min_val
	dataval = np.float32( (dataval - min_val) / (max_val - min_val) )

	if args.dtest is not None:
		datatest[datatest<min_val] = min_val
		datatest = np.float32( (datatest - min_val) / (max_val - min_val) )

	datatrain_out[datatrain_out<min_val] = min_val
	datatrain_out = np.float32( (datatrain_out - min_val) / (max_val - min_val) )

	dataval_out[dataval_out<min_val] = min_val
	dataval_out = np.float32( (dataval_out - min_val) / (max_val - min_val) )	

	### Save normalisation constants
	max_file = open(os.path.join(out_base_dir,'max_val.bin'),'wb')
	pk.dump(max_val,max_file,pk.HIGHEST_PROTOCOL)        
	max_file.close()

	min_file = open(os.path.join(out_base_dir,'min_val.bin'),'wb')
	pk.dump(min_val,min_file,pk.HIGHEST_PROTOCOL)        
	min_file.close()
	
	### Create mini-batches on training data with data loader
	loadertrain = DataLoader(np.concatenate((datatrain,datatrain_out),axis=1), batch_size=mbatch, shuffle=True, num_workers=nwork)

	### Allocate memory for losses
	nobatch=0   # Count how many mini-batches of size mbatch we created
	for signals in loadertrain:
		nobatch = nobatch+1
	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan

	### Instantiate the network and training objects, and save the intantiated network
	nnet = deepqmri.qmriinterp(nhidden,pdrop,actfoo).cpu()                       # Instantiate neural network
	nnetloss = nn.MSELoss()                                                      # Loss: L2 norm (mean squared error, Gaussian noise)
	nnetopt = torch.optim.Adam(nnet.parameters(), lr=lrate)                      # Network trained with ADAM optimiser
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch0_net.pth') )    # Save network at epoch 0 (i.e. at initialisation)
	nnet_file = open(os.path.join(out_base_dir,'epoch0_net.bin'),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()

	### Run training
	# Loop over epochs
	loss_val_prev = np.inf
	for epoch in range(noepoch):
	    
		print('        EPOCH   {}/{}'.format(epoch+1,noepoch))
		print('')

		# Loop over mini-batches for at a fixed epoch
		minibatch_id = 0
		for signals in loadertrain:


			# Pass the mini-batch through the network and store the training loss
			output = nnet( Tensor(signals[:,0:nmeas_in]) )   # Pass input measurements through net and estimate output signals  
			lossmeas_train = nnetloss(Tensor(output), Tensor(signals[:,nmeas_in:nmeas_in+nmeas_out]))  # Training loss 

			# Back propagation
			nnetopt.zero_grad()               # Evaluate loss gradient with respect to network parameters at the output layer
			lossmeas_train.backward()         # Backpropage the loss gradient through previous layers
			nnetopt.step()                    # Update network parameters
		
			# Store loss for the current mini-batch of training
			losstrain[epoch,minibatch_id] = Tensor.numpy(lossmeas_train.data)

			# Update mini-batch counter
			minibatch_id = minibatch_id + 1
		
		
		# Run validation
		nnet.eval()   # Set network to evaluation mode (deactivates dropout)
		dataval_nnet = nnet( Tensor(dataval) )              # Output of full network (predicted MRI signals)
		lossmeas_val = nnetloss( Tensor(dataval_nnet) , Tensor(dataval_out) ) # Validation loss
		# Store validation loss
		lossval[epoch,0] = Tensor.numpy(lossmeas_val.data)

		###### ME QUEDO AQUI

		# Save trained network at current epoch if validation loss has decreased
		if(Tensor.numpy(lossmeas_val.data)<=loss_val_prev):
			print('             ... validation loss has decreased. Saving net...')
			# Save network
			torch.save( nnet.state_dict(), os.path.join(out_base_dir,'lossvalmin_net.pth') )
			nnet_file = open(os.path.join(out_base_dir,'lossvalmin_net.bin'),'wb')
			pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
			nnet_file.close()
			# Save information on the epoch
			nnet_text = open(os.path.join(out_base_dir,'lossvalmin.info'),'w')
			nnet_text.write('Epoch {} (indices starting from 0)'.format(epoch));
			nnet_text.close();
			# Update value of best validation loss so far
			loss_val_prev = Tensor.numpy(lossmeas_val.data)
			# Save predicted validation signals
			dataval_nnet = dataval_nnet.detach().numpy()
			dataval_nnet = min_val + (max_val - min_val)*dataval_nnet
			dataval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigval.bin'),'wb')
			pk.dump(dataval_nnet,dataval_nnet_file,pk.HIGHEST_PROTOCOL)      
			dataval_nnet_file.close()

			# Analyse test data if provided
			if args.dtest is not None:
				# Get neuronal activations as well as predicted test tissue parameters and test MRI signals 
				datatest_nnet = nnet( Tensor(datatest) )              # Output of full network (predicted MRI signals)
				# Save predicted test signals
				datatest_nnet = datatest_nnet.detach().numpy()
				datatest_nnet = min_val + (max_val - min_val)*datatest_nnet
				datatest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigtest.bin'),'wb')
				pk.dump(datatest_nnet,datatest_nnet_file,pk.HIGHEST_PROTOCOL)      
				datatest_nnet_file.close()

		# Set network back to training mode
		nnet.train()

		# Print some information
		print('')
		print('             TRAINING INFO:')
		print('             Trainig loss: {:.12f}; validation loss: {:.12f}'.format(Tensor.numpy(lossmeas_train.data), Tensor.numpy(lossmeas_val.data)) )
		print('')

	# Save the final network
	nnet.eval()
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch{}_net.pth'.format(noepoch)) )
	nnet_file = open(os.path.join(out_base_dir,'epoch{}_net.bin'.format(noepoch)),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()

	# Save the training and validation loss
	losstrain_file = open(os.path.join(out_base_dir,'losstrain.bin'),'wb')
	pk.dump(losstrain,losstrain_file,pk.HIGHEST_PROTOCOL)      
	losstrain_file.close()

	lossval_file = open(os.path.join(out_base_dir,'lossval.bin'),'wb')
	pk.dump(lossval,lossval_file,pk.HIGHEST_PROTOCOL)      
	lossval_file.close()

