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
from numpy import matlib
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
	parser = argparse.ArgumentParser(description='This program trains a qMRI-net for quantitative MRI parameter estimation. A qMRI-Nnet enables voxel-by-voxel estimation of microstructural properties from sets of MRI images aacquired by varying the MRI sequence parameters.  Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')
	parser.add_argument('data_train', help='path to a pickle binary file storing the input training data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('data_val', help='path to a pickle binary file storing the validation data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('mri_model', help='string indicating the MRI model to fit (choose among: "pr_hybriddwi" for prostate hybrid diffusion-relaxometry imaging; "br_sirsmdt" for brain saturation recovery diffusion tensor on spherical mean signals). Tissue parameters will be: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0')
	parser.add_argument('mri_prot', help='path to text file storing the MRI protocol. For model "pr_hybriddwi" it must contain a matrix where the 1st row stores b-values in s/mm^2, while 2nd row echo times in ms; for model "br_sirsmdt" it must contain a matrix where the 1st row stores preparation times (saturation-inversion delay) in ms, the 2nd row inversion times (inversion-excitation delay) in ms, the 3rd row b-values in s/mm^2. For a pure inversion recovery (i.e. no saturation pulse), use a very large number for the saturation-inversion delay (at least 5 times the maximum expected T1). Different entries should be separated by spaces')
	parser.add_argument('out_base', help='base name of output directory (a string built with the network parameters will be added to the base). The output directory will contain the following output files: ** losstrain.bin, pickle binary storing the training loss as a numpy matrix (shape: epoch x batch); ** lossval.bin, pickle binary storing the validation loss as a numpy matrix (shape: epoch x 1); ** nnet_epoch0.bin, pickle binary storing the qMRI-net at initialisation; ** nnet_epoch0.pth, Pytorch binary storing the qMRI-net at initialisation; ** nnet_epoch<FINAL_EPOCH>.bin, pickle binary storing the qMRI-net at the final epoch; ** nnet_lossvalmin.bin, pickle binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); * nnet_lossvalmin.pth, Pytorch binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_sigval.bin, prediction of the validation signals (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_tissueval.bin, prediction of tissue parameters from validation signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_neuronval.bin, output neuron activations for validation signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin.info, text file reporting information regarding the epoch with the lowest validation loss; ** lossval_min.txt, miniimum validation loss;  ** nnet_lossvalmin_sigtest.bin, prediction of the test signals  (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information), if those signals are provided; ** nnet_lossvalmin_tissuetest.bin, prediction of tissue parameters from test signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information) if test signals are provided; ** nnet_lossvalmin_neurontest.bin, output neuron activations for test signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information) if test signals are provided')
	parser.add_argument('--nn', metavar='<list>', help='array storing the number of hidden neurons, separated by hyphens (example: 30-15-8). The first number (input neurons) must equal the number of measurements in the protocol (Nmeas); the last number (output neurons) must equal the number of parameters in the model (Npar, 9 for model "pr_hybriddwi", 4 for model "br_sirsmdt"). Default: Nmeas-(Npar + (Nmeas minus Npar))/2-Npar, where Nmeas is the number of MRI measurements and Npar is the number of tissue parameters for the signal model to fit.')
	parser.add_argument('--pdrop', metavar='<value>', default='0.0', help='dropout probability in each layer of the neural network. Default: 0.0')
	parser.add_argument('--noepoch', metavar='<value>', default='500', help='number of epochs used for training. Default: 500')
	parser.add_argument('--lrate', metavar='<value>', default='0.001', help='learning rate. Default: 0.001')
	parser.add_argument('--mbatch', metavar='<value>', help='number of voxels in each training mini-batch. Default: 1/10 of the total number of training voxels (minimum: 2 voxels)')
	parser.add_argument('--seed', metavar='<value>', default='19102018', help='integer used as a seed for Numpy and PyTorch random number generators. Default: 19102018')
	parser.add_argument('--nwork', metavar='<value>', default='0', help='number of workers for data loader. Default: 0')
	parser.add_argument('--dtest', metavar='<file>', help='path to an option input pickle binary file storing the test data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('--parmin', metavar='<value>', help='list of lower bounds of tissue parameters. Entries corresponding to different parameters should be separated by a comma (for example: 0.5,0.2,250,0.5 for model br_sirsmdt). Tissue parameters are: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0. If not specified, default tissue parameter ranges are used.')
	parser.add_argument('--parmax', metavar='<value>', help='list of upper bounds of tissue parameters. Entries corresponding to different parameters should be separated by a comma (for example: 2.4,0.9,3000,5.0 for model br_sirsmdt). Tissue parameters are: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0. If not specified, default tissue parameter ranges are used.')
	args = parser.parse_args()

	### Get some of the inputs
	pdrop = float(args.pdrop)
	noepoch = int(args.noepoch)
	lrate = float(args.lrate)
	seed = int(args.seed)
	nwork = int(args.nwork)
	mrimodel = args.mri_model

	### Print some information
	print('')
	print('')
	print('********************************************************************')
	print('                 TRAIN A qMRI-NET (qmrisig CLASS)                   ')
	print('********************************************************************')
	print('')
	print('** Input training data: {}'.format(args.data_train))
	print('** Input validation data: {}'.format(args.data_val))
	if args.dtest is not None:
		print('** Input test data: {}'.format(args.dtest))

	### Load training data
	fh = open(args.data_train,'rb')
	datatrain = np.float32(pk.load(fh))
	nmeas_train = datatrain.shape[1]
	nmeas_out = nmeas_train 
	fh.close()

	### Load validation data
	fh = open(args.data_val,'rb')
	dataval = np.float32(pk.load(fh))
	fh.close()
	if dataval.shape[1]!=datatrain.shape[1]:
		raise RuntimeError('the number of MRI measurements in the validation set differs from the training set!')		

	### Load test MRI signals
	if args.dtest is not None:
		fh = open(args.dtest,'rb')
		datatest = np.float32(pk.load(fh))
		fh.close()
		if datatest.shape[1]!=datatrain.shape[1]:
			raise RuntimeError('the number of MRI measurements in the test set differs from the training set!')		

	### Get number of mini-batches
	if args.mbatch is None:
		mbatch = int(float(datatrain.shape[0]) / 10.0) # Default: 1/10 of the total number of training voxels
	else:
		mbatch = int(args.mbatch)
		if (mbatch>datatrain.shape[0]):
			mbatch = datatrain.shape[0]
		if(mbatch<2):
			mbatch = int(2)

	### Load MRI protocol
	try:
		mriprot = np.loadtxt(args.mri_prot)
	except:
		raise RuntimeError('the format of the MRI protocol is not understood!')
		
	### Check that MRI model exists
	if ( (mrimodel!='pr_hybriddwi') and (mrimodel!='br_sirsmdt') ):
		raise RuntimeError('the chosen MRI model is not implemented. Sorry!')
	if (mrimodel=='pr_hybriddwi'):
		s0idx = 8
	elif (mrimodel=='br_sirsmdt'):
		s0idx = 3

	### Get specifics for hidden layers
	if args.nn is None:

		if (mrimodel=='pr_hybriddwi'): 
			npars = 9
		elif (mrimodel!='br_sirsmdt'):
			npars = 4
		else:
			raise RuntimeError('the chosen MRI model is not implemented. Sorry!')


		nhidden = np.array([int(nmeas_train) , int(float(npars)+0.5*( float(nmeas_train) - float(npars))) , int(npars)])
		nhidden_str = '{}-{}-{}'.format( int(nmeas_train) , int(float(npars)+0.5*( float(nmeas_train) - float(npars))) , int(npars)  )

	else:
		nhidden = (args.nn).split('-')
		nhidden = np.array( list(map( int,nhidden )) )
		nhidden_str = args.nn
		
	### Get optional user-defined bounds for tissue parameters
	if (args.parmin is not None) or (args.parmax is not None):
		
		if (args.parmin is not None) and (args.parmax is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
		
		if (args.parmax is not None) and (args.parmin is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
					
		# Lower bound
		pminbound = (args.parmin).split(',')
		pminbound = np.array( list(map( float, pminbound )) )
		
		# Upper bound
		pmaxbound = (args.parmax).split(',')
		pmaxbound = np.array( list(map( float, pmaxbound )) )


	### Create output base name
	out_base_dir = '{}_nhidden{}_pdrop{}_noepoch{}_lr{}_mbatch{}_seed{}'.format(args.out_base,nhidden_str,pdrop,noepoch,lrate,mbatch,seed)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir)

	### Print some more information
	print('** Output directory: {}'.format(out_base_dir))
	print('')
	print('')
	print('PARAMETERS')
	print('')
	print('** Hidden neurons: {}'.format(nhidden))
	print('** Dropout probability: {}'.format(pdrop))
	print('** Number of epochs: {}'.format(noepoch))
	print('** Learning rate: {}'.format(lrate))
	print('** Number of voxels in a mini-batch: {}'.format(mbatch))
	print('** Seed: {}'.format(seed))
	print('** Number of workers for data loader: {}'.format(nwork))


	### Set random seeds
	np.random.seed(seed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(seed)    # Random seed for reproducibility: PyTorch

	### Normalise data
	max_val_train = np.transpose( matlib.repmat(np.max(datatrain,axis=1),nmeas_train,1) )
	datatrain = np.float32( datatrain / max_val_train )

	max_val_val = np.transpose( matlib.repmat(np.max(dataval,axis=1),nmeas_train,1) )
	dataval = np.float32( dataval / max_val_val )

	if args.dtest is not None:
		max_val_test = np.transpose( matlib.repmat(np.max(datatest,axis=1),nmeas_train,1) )
		datatest = np.float32( datatest / max_val_test )	
	
	### Create mini-batches on training data with data loader
	loadertrain = DataLoader(datatrain, batch_size=mbatch, shuffle=True, num_workers=nwork)

	### Allocate memory for losses
	nobatch=0   # Count how many mini-batches of size mbatch we created
	for signals in loadertrain:
		nobatch = nobatch+1
	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan

	### Instantiate the network and training objects, and save the intantiated network
	nnet = deepqmri.qmrisig(nhidden,pdrop,mrimodel,mriprot).cpu()                # Instantiate neural network
	if (args.parmin is not None) or (args.parmax is not None):
		nnet.changelim(pminbound,pmaxbound)                                  # Change tissue parameter ranges
	print('** Tissue parameter names: {}'.format(nnet.param_name))
	print('** Tissue parameter lower bounds: {}'.format(nnet.param_min))	
	print('** Tissue parameter upper bounds: {}'.format(nnet.param_max))	
	print('')
	print('')	
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
			output = nnet( Tensor(signals) )        # Pass MRI measurements through net and get estimates of MRI signals  
			lossmeas_train = nnetloss(Tensor(output), Tensor(signals))  # Training loss 

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
		tissueval_nnet = nnet.getparams( Tensor(dataval) )  # Estimated tissue parameters
		neuronval_nnet = nnet.getneurons( Tensor(dataval) ) # Output neuron activations
		lossmeas_val = nnetloss( dataval_nnet , Tensor(dataval) ) # Validation loss
		dataval_nnet = dataval_nnet.detach().numpy()
		max_val_val_out = np.transpose( matlib.repmat(np.max(dataval_nnet,axis=1),nmeas_train,1) )
		
		# Store validation loss
		lossval[epoch,0] = Tensor.numpy(lossmeas_val.data)

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
			dataval_nnet = (max_val_val/max_val_val_out)*dataval_nnet   # Rescale signals
			dataval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigval.bin'),'wb')
			pk.dump(dataval_nnet,dataval_nnet_file,pk.HIGHEST_PROTOCOL)      
			dataval_nnet_file.close()
			# Save predicted validation tissue parameters 
			tissueval_nnet = tissueval_nnet.detach().numpy()
			tissueval_nnet[:,s0idx] = (max_val_val[:,0]/max_val_val_out[:,0])*tissueval_nnet[:,s0idx] # # Rescale s0 (any column of would work)
			tissueval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissueval.bin'),'wb')
			pk.dump(tissueval_nnet,tissueval_nnet_file,pk.HIGHEST_PROTOCOL)      
			tissueval_nnet_file.close()
			# Save output validation neuron activations 
			neuronval_nnet = neuronval_nnet.detach().numpy()
			neuronval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_neuronval.bin'),'wb')
			pk.dump(neuronval_nnet,neuronval_nnet_file,pk.HIGHEST_PROTOCOL)      
			neuronval_nnet_file.close()

			# Analyse test data if provided
			if args.dtest is not None:
				# Get neuronal activations as well as predicted test tissue parameters and test MRI signals 
				datatest_nnet = nnet( Tensor(datatest) )              # Output of full network (predicted MRI signals)
				datatest_nnet = datatest_nnet.detach().numpy()
				max_val_test_out = np.transpose( matlib.repmat(np.max(datatest_nnet,axis=1),nmeas_train,1) )
				tissuetest_nnet = nnet.getparams( Tensor(datatest) )  # Estimated tissue parameters
				neurontest_nnet = nnet.getneurons( Tensor(datatest) ) # Output neuron activations
				# Save predicted test signals
				datatest_nnet = (max_val_test/max_val_test_out)*datatest_nnet # Rescale signals
				datatest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigtest.bin'),'wb')
				pk.dump(datatest_nnet,datatest_nnet_file,pk.HIGHEST_PROTOCOL)      
				datatest_nnet_file.close()
				# Save predicted test parameters 
				tissuetest_nnet = tissuetest_nnet.detach().numpy()
				tissuetest_nnet[:,s0idx] = (max_val_test[:,0]/max_val_test_out[:,0])*tissuetest_nnet[:,s0idx] # Rescale s0 (any column works)
				tissuetest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissuetest.bin'),'wb')
				pk.dump(tissuetest_nnet,tissuetest_nnet_file,pk.HIGHEST_PROTOCOL)        
				tissuetest_nnet_file.close()
				# Save output test neuron activations 
				neurontest_nnet = neurontest_nnet.detach().numpy()
				neurontest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_neurontest.bin'),'wb')
				pk.dump(neurontest_nnet,neurontest_nnet_file,pk.HIGHEST_PROTOCOL)      
				neurontest_nnet_file.close()

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
	np.savetxt(os.path.join(out_base_dir,'lossval_min.txt'), [np.nanmin(lossval)], fmt='%.12f', delimiter=' ')

