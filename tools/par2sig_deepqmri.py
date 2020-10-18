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
import nibabel as nib
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
	parser = argparse.ArgumentParser(description='This program synthesises MRI signals given a set of tissue parameters previously estimated with a qMRI-Net and given an input MRI protocol. Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')	
	parser.add_argument('par_in', help='path to file storing estimated tissue parameters (it can be a 4D NIFTI or a python pickle binary file; in the latter case, voxels are stored along rows and measurements along columns)')
	parser.add_argument('mri_prot', help='path to text file storing the MRI protocol. For model "pr_hybriddwi" it must contain a matrix where the 1st row stores b-values in s/mm^2, while 2nd row echo times in ms; for model "br_sirsmdt" it must contain a matrix where the 1st row stores preparation times (saturation-inversion delay) in ms, the 2nd row inversion times (inversion-excitation delay) in ms, the 3rd row b-values in s/mm^2. For a pure inversion recovery (i.e. no saturation pulse), use a very large number for the saturation-inversion delay (at least 5 times the maximum expected T1). Different entries should be separated by spaces')
	parser.add_argument('mri_model', help='string indicating the MRI model to fit (choose among: "pr_hybriddwi" for prostate hybrid diffusion-relaxometry imaging; "br_sirsmdt" for brain saturation recovery diffusion tensor on spherical mean signals). Tissue parameters must be in this order: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0')
	parser.add_argument('sig_out', help='path of the output file storing the MRI signal obtained by applying the forward MRI signal model to the input tissue parameters given the input MRI protocol (it will have the same format as the mandatory input par_in, i.e. NIFTI if sig_in is a NIFTI, pickle binary otherwise)')
	parser.add_argument('--mask', metavar='<nifti>', help='path to a mask flagging with 1 voxels to analysise and with 0 voxels to ignore (the output NIFTI will contain 0 in voxels to ignore; note that the mask is ignored if par_in is a pickle binary file)')
	args = parser.parse_args()

	### Print some information
	print('')
	print('**************************************************************************************')
	print('           GET MRI SIGNALS FROM TISSUE PARAMETERS ESTIMATED WITH A qMRI-NET           ')
	print('**************************************************************************************')
	print('')
	print('** Called on input tissue parameters file: {}'.format(args.par_in))
	print('** MRI protocol file: {}'.format(args.mri_prot))
	print('** MRI signal model: {}'.format(args.mri_model))
	print('** Output file storing predicted MRI signal: {}'.format(args.sig_out))
	print('')
	print('')	

	### Load input data
	print('')
	print('      ... loading data ...')

	# Try NIFTI first
	try:
		# Load NIFTI
		par_obj = nib.load(args.par_in)
		par_header = par_obj.header
		par_affine = par_header.get_best_affine()
		par_data = par_obj.get_data()
		par_dims = par_data.shape
		imgsize = par_data.shape
		imgsize = np.array(imgsize)
		par_data = np.array(par_data,'float32')
		isnifti = True

		# Check that NIFTI is 4D
		if imgsize.size!=4:
			print('')
			raise RuntimeError('ERROR: the input 4D NIFTI file {} is not actually not 4D. Exiting with 1...'.format(args.sig_in))

		# Check whether a fitting mask has been provided
		if isinstance(args.mask, str)==1:
			# A mask for qMRI-net has been provided
			mask_obj = nib.load(args.mask)
			mask_dims = mask_obj.shape		
			mask_header = mask_obj.header
			mask_affine = mask_header.get_best_affine()			
			# Make sure the mask is a 3D file
			mask_data = mask_obj.get_data()
			masksize = mask_data.shape
			masksize = np.array(masksize)
			
			if masksize.size!=3:
				print('')
				print('WARNING: the mask file {} is not a 3D Nifti file. Ignoring mask...'.format(mask_nifti))				 
				print('')
				mask_data = np.ones(imgsize[0:3],'float64')
			elif ( (np.sum(par_affine==mask_affine)!=16) or (par_dims[0]!=mask_dims[0]) or (par_dims[1]!=mask_dims[1]) or (par_dims[2]!=mask_dims[2]) ):
				print('')
				print('WARNING: the geometry of the mask file {} does not match that of the input data. Ignoring mask...'.format(args.mask))					 
				print('')
				mask_data = np.ones(imgsize[0:3],'float64')
			else:
				mask_data = np.array(mask_data,'float64')
				# Make sure mask data is a numpy array
				mask_data[mask_data>0] = 1
				mask_data[mask_data<=0] = 0

		else:
			# A mask for fitting has not been provided
			mask_data = np.ones(imgsize[0:3],'float64')		


		
	# NIFTI has not worked
	except:

		# Try pickle binary then
		try:
			hin = open(args.par_in,'rb')
			par_data = pk.load(hin)
			hin.close()
			isnifti = False

		# Unknown input data format!
		except:
			raise RuntimeError('the format of the input data to analyse is not understood!')


	### Load MRI protocol
	try:
		mriprot = np.loadtxt(args.mri_prot)
		Nmeas = mriprot.shape[1]  # Number of MRI measurements
	except:
		raise RuntimeError('the format of the MRI protocol is not understood!')

	### Instantiate a qMRI-Net to apply the forward model to the tissue parameters
	mrimodel = args.mri_model
	if ( (mrimodel!='pr_hybriddwi') and (mrimodel!='br_sirsmdt') ):
		raise RuntimeError('the chosen MRI model is not implemented. Sorry!')
	if mrimodel=='pr_hybriddwi':
		qnet = deepqmri.qmrisig([Nmeas,9],0.0,2.0,'pr_hybriddwi',mriprot)
	elif mrimodel=='br_sirsmdt':
		qnet = deepqmri.qmrisig([Nmeas,4],0.0,2.0,'br_sirsmdt',mriprot)

	### Get signals given the input tissue parameters  
	print('')
	print('     ... calculating MRI signals given the tissue parameters ...')

	# NIFTI
	if(isnifti==True):	

		# Allocate variable to store output MRI signals
		sigout = np.zeros( (imgsize[0], imgsize[1], imgsize[2], Nmeas)  )     # Output MRI signals

		# Loop through voxels to estimate tissue parameters with a trained net
		for xx in range(0, imgsize[0]):
			for yy in range(0, imgsize[1]):
				for zz in range(0, imgsize[2]):

					# Get voxel within mask
					if(mask_data[xx,yy,zz]==1):

						# Get tissue parameters
						mypars = par_data[xx,yy,zz,:]
						# Get MRI signals from tissue parameters
						mysigs = qnet.getsignals(Tensor(mypars))
						mysigs = mysigs.detach().numpy()
						for nn in range(0, Nmeas): 
							sigout[xx,yy,zz,nn] = mysigs[nn]
			
		
		# Save the predicted NIFTI file as output
		print('')
		print('     ... saving output MRI signals as 4D NIFTI ...')
		buffer_header = par_obj.header
		buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
		sigout_obj = nib.Nifti1Image(sigout,par_obj.affine,buffer_header)
		try:
			nib.save(sigout_obj, args.sig_out)
		except:
			raise RuntimeError('the destination folder may not exist or you may lack permissions to write there!')

		# Done
		print('')
		print('     Done!')
		print('')
		sys.exit(0)

	# Python pickle binaries
	else:

		# Get MRI signals from tissue parameters
		sigout = qnet.getsignals(Tensor(par_data))              # Tissue parameters
		sigout = sigout.detach().numpy()

		# Save estimated tissue parameters as well as predicted MRI signals/output neuron activations if requested
		try:
			print('')
			print('     ... saving output MRI signals as a pickle binary file ...')
			out_file = open(args.sig_out,'wb')
			pk.dump(sigout,out_file,pk.HIGHEST_PROTOCOL)      
			out_file.close()
		except:
			raise RuntimeError('the destination folder may not exist or you may lack permissions to write there!')

		# Done
		print('')
		print('     Done!')
		print('')
		sys.exit(0)



