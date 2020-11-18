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
	parser = argparse.ArgumentParser(description='This program uses a trained qMRI-net (class qmriinterp) to map a quantitative MRI signal acquired according to a first protocol to a resampled version of the the signal corresponding to a second, target protocol. Processing is preformed on a voxel-by-voxel basis. It requires as input a trained qMRI-net, an MRI scan to analyse (either a 4D NIFTI or a python pickle binary file storing a matrix with voxels along rows and measurements along columns), information on signal normalisation and an optional mask (if the input scan is NIFTI). It outputs the resampled MRI signals in either NIFTI or python pickle binary format (the same format as that of the input data to analyse is used). Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')	
	parser.add_argument('sig_in', help='path to file storing a multi-contrast MRI scan to analyse (it can be a 4D NIFTI or a python pickle binary file; in the latter case, voxels are stored along rows and measurements along columns)')
	parser.add_argument('sig_out', help='path of output file storing the output MRI scan resampled according to the target protocol (it will have the same format as the mandatory input sig_in, i.e. NIFTI if sig_in is a NIFTI, pickle binary otherwise)')
	parser.add_argument('qmrinet_file', help='path to a pickle binary file containing a trained qMRI-net (class qmriinterp)')
	parser.add_argument('max_file', help='path to a pickle binary file containing the maximum signal for normalisation (max_val.bin file provided by traininterp_deepqmri.py)')
	parser.add_argument('min_file', help='path to a pickle binary file containing the minimum signal for normalisation (min_val.bin file provided by traininterp_deepqmri.py)')
	parser.add_argument('--mask', metavar='<nifti>', help='path to a mask flagging with 1 voxels to analysise and with 0 voxels to ignore (the output NIFTI will contain 0 in voxels to ignore; note that the mask is ignored if sig_in is a pickle binary file)')

	args = parser.parse_args()

	### Print some information
	print('')
	print('********************************************************************')
	print('       RESAMPLE MRI PROTOCOL WITH qMRI-NET (qmriinterp CLASS)       ')
	print('********************************************************************')
	print('')
	print('** Called on input MRI measurement file: {}'.format(args.sig_in))
	print('** qMRI-net file (class qmriinterp): {}'.format(args.qmrinet_file))
	print('** Output file storing the resampled MRI measurements: {}'.format(args.sig_out))
	print('')
	print('')	

	### Load input measurements
	print('')
	print('      ... loading data ...')

	# Try NIFTI first
	try:
		# Load NIFTI
		sin_obj = nib.load(args.sig_in)
		sin_header = sin_obj.header
		sin_affine = sin_header.get_best_affine()
		sin_data = sin_obj.get_fdata()
		sin_dims = sin_data.shape
		imgsize = sin_data.shape
		imgsize = np.array(imgsize)
		sin_data = np.array(sin_data,'float32')
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
			mask_data = mask_obj.get_fdata()
			masksize = mask_data.shape
			masksize = np.array(masksize)
			
			if masksize.size!=3:
				print('')
				print('WARNING: the mask file {} is not a 3D Nifti file. Ignoring mask...'.format(mask_nifti))				 
				print('')
				mask_data = np.ones(imgsize[0:3],'float64')
			elif ( (np.sum(sin_affine==mask_affine)!=16) or (sin_dims[0]!=mask_dims[0]) or (sin_dims[1]!=mask_dims[1]) or (sin_dims[2]!=mask_dims[2]) ):
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
			hin = open(args.sig_in,'rb')
			sin_data = pk.load(hin)
			hin.close()
			isnifti = False

		# Unknown input data format!
		except:
			raise RuntimeError('the format of the input data to analyse is not understood!')

						 

	#### Load qMRI-net in evaluation mode
	try:	
		h = open(args.qmrinet_file,'rb') 
		net = pk.load(h)
		h.close()
	except:
		raise RuntimeError('the format of the trained qMRI-Net is not understood!')
	net.eval()


	### Load normalisation factors
	try:
		h = open(args.max_file,'rb') 
		max_val = pk.load(h)
		h.close()

		h = open(args.min_file,'rb') 
		min_val = pk.load(h)
		h.close()
	except:
		raise RuntimeError('the format of the signal normalisation factors is not understood!')


	### Resample MRI signals 
	print('')
	print('     ... predicting MRI signals with qMRI-net ...')

	# NIFTI
	if(isnifti==True):


		# Get normalisation factors
		smax = np.float32(max_val)
		smin = np.float32(min_val)

		# Number of output measurements
		nneurnet = net.nneurons
		nmeas_out = nneurnet[-1] 

		# Allocate variables to store output results
		sigout = np.zeros( (imgsize[0], imgsize[1], imgsize[2], nmeas_out)  )        # Output resampled MRI signals

		# Loop through voxels to estimate tissue parameters with a trained net
		for xx in range(0, imgsize[0]):
			for yy in range(0, imgsize[1]):
				for zz in range(0, imgsize[2]):

					# Get voxel within mask
					if(mask_data[xx,yy,zz]==1):

						# Get qMRI signal
						myvoxel = sin_data[xx,yy,zz,:]

						# Normalise qMRI signal
						myvoxel[myvoxel<smin] = smin
						myvoxel = np.float32( (myvoxel - smin) / (smax - smin) )

						# Pass voxel through qMRI-net
						myvoxel_sigs = net(Tensor(myvoxel))
						myvoxel_sigs = myvoxel_sigs.detach().numpy()

						# Un-normalise
						myvoxel_sigs = smin + (smax - smin)*(myvoxel_sigs)

						# Store resampled voxel
						for mm in range(0, nmeas_out):
							sigout[xx,yy,zz,mm] = myvoxel_sigs[mm] 

			
		
		# Save the predicted NIFTI file as output
		print('')
		print('     ... saving resampled MRI signals as 4D NIFTI ...')
		buffer_header = sin_obj.header
		buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
		sigout_obj = nib.Nifti1Image(sigout,sin_obj.affine,buffer_header)
		try:
			nib.save(sigout_obj, args.sig_out)
		except:
			raise RuntimeError('the destination folder may not exist or you may lack permissions to write there!')

		print('')
		print('     Done!')
		print('')
		sys.exit(0)


	# Python pickle binaries
	else:

		# Get normalisation factors
		smax = np.float32(max_val)
		smin = np.float32(min_val)

		# Normalise qMRI signals
		sin_data[sin_data<smin] = smin
		sin_data = np.float32( (sin_data - smin) / (smax - smin) )

		# Pass signals through qMRI-net as a whole batch to get resampled MRI signals
		sigout = net(Tensor(sin_data))                # Predicted MRI signals 
		sigout = sigout.detach().numpy()

		# Un-normalise
		sigout = smin + (smax - smin)*sigout

		# Save estimated MRI signals
		try:
			print('')
			print('     ... saving resampled MRI signals as a pickle binary file ...')
			sout_file = open(args.sig_out,'wb')
			pk.dump(sigout,sout_file,pk.HIGHEST_PROTOCOL)      
			sout_file.close()
		except:
			raise RuntimeError('the destination folder may not exist or you may lack permissions to write there!')

		print('')
		print('     Done!')
		print('')
		sys.exit(0)




