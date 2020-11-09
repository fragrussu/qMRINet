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
import glob as gb
import pickle as pk
import nibabel as nib
import scipy.io as sio


if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program extract voxels within tissues of interest of various subjects and creates training and validation sets to train a qMRI-net. Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')
	parser.add_argument('data_dir', help='path to a folder storing 4D NIFTI files with multi-contrast MRI scans from the training cohort. Scans should be named like: scan_1.nii, ..., scan_9.nii, scan_10.nii, ..., scan_100.nii, etc (note that format .nii.gz would work as well).')
	parser.add_argument('data_train', help='path of the pickle binary output file storing the training data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('data_val', help='path of the pickle binary output file storing the validation data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('--usemask', metavar='<value>', default='0', help='flag indicating whether tissue should be extracted only from selected voxels within specificed masks or not (default 0, meaning no masks). If set to 1 (meaning masks are required), a mask per MRI scan should be provided as a 3D binary NIFTI file: mask_1.nii referring to scan_1.nii, ..., mask_9.nii referring to scan_9.nii, mask_10.nii referring to scan_10.nii, ..., mask_100.nii referring to scan_100.nii, etc  (note that format .nii.gz would work as well). If the flag is set to any other value different from 1 masks will be ignored.')
	parser.add_argument('--valratio', metavar='<value>', default='0.2', help='fraction of total voxels to be used for validation, defined in the range [0.01; 0.99] (default 0.2, implying that 20%% of the voxels will be stored as validation set and 80%% as training set).')
	parser.add_argument('--matlab_train', metavar='<value>' , help='path of an optional output binary file storing the training data in MATLAB format (The MathWorks, Inc., Natick, Massachusetts, USA)')
	parser.add_argument('--matlab_val', metavar='<value>' , help='path of an optional output binary file storing the training data in MATLAB format (The MathWorks, Inc., Natick, Massachusetts, USA)')
	args = parser.parse_args()

	print('')
	print('')
	print('********************************************************************')
	print('              EXTRACT MRI SIGNALS FOR qMRI-NET TRAINING             ')
	print('********************************************************************')
	print('')
	print('** Input folder: {}'.format(args.data_dir))
	print('** Ouput training signal matrix (pickle binary): {}'.format(args.data_train))
	if(args.matlab_train!=None):
		print('                                                 {} (MATLAB format)'.format(args.matlab_train))
	print('** Ouput validation signal matrix (pickle binary): {}'.format(args.data_val))
	if(args.matlab_val!=None):
		print('                                                 {} (MATLAB format)'.format(args.matlab_val))
	print('')
	print('')

	### Get optional inputs
	usemask = int(args.usemask)
	if( (usemask!=0) and (usemask!=1) ):
		print('WARNING! The usemask flag must be either 0 or 1 (you set {} instead). Masks will be ignored!'.format(usemask))
		usemask = 0
	
	valratio = float(args.valratio)
	if( (valratio<0.01) or (valratio > 0.99) ):
		raise RuntimeError('the fraction of total voxels to be used for validation must be between 0.01 and 0.99 (i.e. between 1% and 99%). You set {} instead!'.format(valratio))

	### Get list of input 4D NIFTI scans
	mrilist = gb.glob(os.path.join(args.data_dir,'scan_*nii*'))
	nfiles = len(mrilist)
	if(nfiles==0):
		raise RuntimeError('no MRI files in the input folder {} detected! Files should be named as scan_1.nii, scan_2.nii, ...'.format(args.data_dir))

	### Load all NIFTI files and create signal matrices
	print ('')
	print('  ..... loading input scans, please wait:')
	for nn in range(0,nfiles):

		# Load NIFTI file and get information on image size, header etc
		print('                                        scan {}/{}'.format(nn+1,nfiles))
		myscan_obj = nib.load(mrilist[nn])
		myscan_data = myscan_obj.get_data()
		imgsize = np.array(myscan_data.shape)
		if imgsize.size!=4:
			raise RuntimeError('the scan {} is not a 4D NIFTI.'.format(mrilist[nn]))
		if(nn>0):
			if(imgsize[3]!=nmeas):
				raise RuntimeError('the scan {} has a different number of the previous ones.'.format(mrilist[nn]))
		nmeas = imgsize[3]
		if(nn==0):
			datacat = np.zeros((0,nmeas))
	

		# Load mask (if a mask is requested) or create a masks with 1 everywhere (if a mask is not requested)
		if (usemask==1):

			# Sort out file names			
			dirname = os.path.dirname(mrilist[nn])
			filename = os.path.basename(mrilist[nn])
			maskname = os.path.join(dirname,'mask_{}'.format(filename[5:len(filename)]))
			
			# Load mask
			mask_obj = nib.load(maskname)
			mask_data = mask_obj.get_data()
			mask_dims = mask_obj.shape		
			mask_header = mask_obj.header
			mask_affine = mask_header.get_best_affine()
			
			# Make sure the mask is a 3D file and its header is consistent with the MRI scan it refers to
			sig_header = myscan_obj.header
			sig_affine = sig_header.get_best_affine()
			sig_dims = myscan_obj.shape
			masksize = mask_data.shape
			masksize = np.array(masksize)
			if masksize.size!=3:
				print('')
				print('  WARNING: the mask file {} is not a 3D Nifti file. Ignoring mask...'.format(maskname))				 
				print('')
				mask_data = 0.0*myscan_data[:,:,:,0] + 1.0
			elif ( (np.sum(sig_affine==mask_affine)!=16) or (sig_dims[0]!=mask_dims[0]) or (sig_dims[1]!=mask_dims[1]) or (sig_dims[2]!=mask_dims[2]) ):
				print('')
				print('  WARNING: the geometry of mask file {} does not match that of the corresponding scan ({}). Ignoring mask...'.format(maskname,mrilist[nn]))					 
				print('')
				mask_data = 0.0*myscan_data[:,:,:,0] + 1.0
			else:
				mask_data = np.array(mask_data,'float64')
				mask_data[mask_data>0] = 1.0
				mask_data[mask_data<=0] = 0.0
		else:
			mask_data = 0.0*myscan_data[:,:,:,0] + 1.0

		
		# Load voxels and stack them as matrices (rows: VOXELS x columns: MEASUREMENTS)
		nvox = len(mask_data[mask_data==1])
		mat = np.zeros((nvox,nmeas))
		vox_count = 0
		for ii in range(imgsize[0]):
			for jj in range(imgsize[1]):
				for kk in range(imgsize[2]):

					if(mask_data[ii,jj,kk]==1):
						mat[vox_count,:] = np.squeeze(myscan_data[ii,jj,kk,:])
						vox_count = vox_count + 1


		
		# Store the new scan and concatenate it with previous scans 
		datacat = np.concatenate((datacat,mat),axis=0)

	### Shuffle randomly voxels
	datasize = datacat.shape[0]
	dataidx = np.arange(datasize)
	np.random.shuffle(dataidx)
	datacat = datacat[dataidx,:]


	### Split signal matrices into data on training and validation sets
	print ('')
	print('  ..... creating training and validation sets (ratio: {}%/{}%), please wait'.format( np.round((1.0 - valratio)*100.0) , np.round(valratio*100.0)  ) )
	ntot = datacat.shape[0]
	nval = np.int(np.round(valratio*ntot))
	idx_val = np.zeros((ntot),dtype=bool)
	pick_val = np.random.choice(ntot, nval, replace=False)
	idx_val[pick_val] = True
	idx_train = ~idx_val

	
	### Save training and validation sets as pickle binary files
	print ('')
	print('  ..... saving output files, please wait')
	my_file = open(args.data_train,'wb')
	pk.dump(datacat[idx_train,:],my_file,pk.HIGHEST_PROTOCOL)
	my_file.close()	

	my_file = open(args.data_val,'wb')
	pk.dump(datacat[idx_val,:],my_file,pk.HIGHEST_PROTOCOL)
	my_file.close()	
		
	### Save training and validation sets as MATLAB binary files if required
	if(args.matlab_train!=None):
		buffer_mat = {}
		buffer_mat['data'] = datacat[idx_train,:]
		sio.savemat(args.matlab_train, buffer_mat)
		del buffer_mat

	if(args.matlab_val!=None):
		buffer_mat = {}
		buffer_mat['data'] = datacat[idx_val,:]
		sio.savemat(args.matlab_val, buffer_mat)
		del buffer_mat


	### Done!
	print ('')
	print('  ..... done!')
	print('')





