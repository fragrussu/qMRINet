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


if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program downsamples a file storing a set of multi-contrast quantitative MRI measurements and outputs the downsampled MRI sub-protocol. Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2019 University College London. All rights reserved.')	
	parser.add_argument('filein', help='path to a 4D NIFTI file or a python pickle binary file storing a multi-contrast MRI measurements to downsample (for pickle binary files: it must store a 2D matrix with voxels along rows, and measurements along columns)')
	parser.add_argument('fileout', help='path of the output file that will store the downsampled MRI measurements (it will have the same format as the input file)')
	parser.add_argument('subset', help='path to a text file containing the indices of the volumes to keep in the output (NOTE: indices MUST start from 0 and range from 0 to no. of measurements - 1)')
	parser.add_argument('--bits', metavar='<value>',  default='64', help='number of bits per voxel in the output file (default 64 bits; choose either 32 or 64 bits)')
	args = parser.parse_args()

	### Print some information
	print('')
	print('********************************************************************')
	print('          EXTRACT A SUB-PROTOCOL FROM A 4D MRI EXPERIMENT           ')
	print('********************************************************************')
	print('')
	print('** Called on file: {}'.format(args.filein))
	print('** Subset file: {}'.format(args.subset))
	print('** Output: {}'.format(args.fileout))
	print('')	

	### Load input data
	print('')
	print('      ... loading data ...')

	## Bits per pixel
	nbit = int(args.bits)
	if( (nbit!=32) and (nbit!=64) ):
		print('')
		print('ERROR: the voxel bit depth must be either 32 or 64. You set {}. Exiting with 1...'.format(args.bits))				 
		print('')
		sys.exit(1)

	## Indices of measurements to keep
	idx = np.loadtxt(args.subset)        # Indices of measurements to keep
	idx = np.array(idx)                  # Convert to array
	idx = idx.astype(int)                # Convert to integer

	## Load MRI scan
	# Try NIFTI first
	try:
		sin_obj = nib.load(args.filein)
		sin_header = sin_obj.header
		sin_affine = sin_header.get_best_affine()
		sin_data = sin_obj.get_data()
		sin_dims = sin_data.shape
		imgsize = sin_data.shape
		imgsize = np.array(imgsize)
		sin_data = np.array(sin_data,'float64')
		if imgsize.size!=4:
			print('')
			raise RuntimeError('ERROR: the input 4D NIFTI file {} is not actually not 4D. Exiting with 1...'.format(args.nifti_in))
		isnifti=True
		Nmeas = imgsize[3] # Total number of MRI measurements
	except:
		# Try pickle binary then
		try:
			hin = open(args.filein,'rb')
			sin_data = pk.load(hin)
			hin.close()
			isnifti = False
			Nmeas = sin_data.shape[1] # Total number of MRI measurements

		# Unknown input data format!
		except:
			raise RuntimeError('the format of the input data to analyse is not understood!')


	### Downsample scan
	try:
		if isnifti==True:
			print('')
			print('      ... downsampling scan: {}/{} measurements'.format(idx.size,sin_data.shape[3])) 
			print('')
			sout_data = sin_data[:,:,:,idx]
		else:
			print('')
			print('      ... downsampling scan: {}/{} measurements'.format(idx.size,sin_data.shape[1])) 
			print('')
			sout_data = sin_data[:,idx]
	except:
		raise RuntimeError('the indices provided for the subset do not match the input MRI measurements!')

		
	### Save downsampled scan as NIFTI
	print('     ... saving output file as 4D NIFTI ...')
	try:
		if isnifti==True:
			buffer_header = sin_obj.header
			if(nbit==64):
				buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
			elif(nbit==32):
				buffer_header.set_data_dtype('float32')   # Make sure we save output files float64, even if input is not
			sout_obj = nib.Nifti1Image(sout_data,sin_obj.affine,buffer_header)
			nib.save(sout_obj, args.fileout)
		else:
			sout_file = open(args.fileout,'wb')
			pk.dump(sout_data,sout_file,pk.HIGHEST_PROTOCOL)
			sout_file.close()
	except:
		raise RuntimeError('the destination folder may not exist or you may lack permissions to write there!')

	print('')
	print('     Done!')
	print('')

	sys.exit(0)

