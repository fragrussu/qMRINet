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
import pickle as pk
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
	parser = argparse.ArgumentParser(description='This program synthesise MRI signals that can be used to train a qMRI-net for quantitative MRI parameter estimation.  Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')
	parser.add_argument('mri_model', help='string indicating the MRI model to fit (choose among: "pr_hybriddwi" for prostate hybrid diffusion-relaxometry imaging; "br_sirsmdt" for brain saturation recovery diffusion tensor on spherical mean signals; "twocompdwite" for a two-compartment diffusion-t2 relaxation model without anisotropy)). Tissue parameters will be: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0; model "twocompdwite", parameters v, Da, t2a, Db, Kb, t2b, s0')
	parser.add_argument('mri_prot', help='path to text file storing the MRI protocol. For model "pr_hybriddwi" and "twocompdwite" it must contain a matrix where the 1st row stores b-values in s/mm^2, while 2nd row echo times in ms; for model "br_sirsmdt" it must contain a matrix where the 1st row stores preparation times (saturation-inversion delay) in ms, the 2nd row inversion times (inversion-excitation delay) in ms, the 3rd row b-values in s/mm^2. For a pure inversion recovery (i.e. no saturation pulse), use a very large number for the saturation-inversion delay (at least 5 times the maximum expected T1). Different entries should be separated by spaces')
	parser.add_argument('out_str', help='base name of output files, to which various strings will be appended: file *_sigtrain.bin will store synthetic training MRI signals as a numpy matrix (rows: voxels; columns: measurements); file *_sigval.bin will store synthetic validation MRI signals as a numpy matrix (rows: voxels; columns: measurements); file *_sigtest.bin will store synthetic test MRI signals as a numpy matrix (rows: voxels; columns: measurements); file *_paramtrain.bin will store tissue parameters corresponding to training MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); file *_paramval.bin will store tissue parameters corresponding to validation MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); file *_paramtest.bin will store tissue parameters corresponding to test MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); file *_sigmatrain.bin will store voxel-wise noise levels for validation voxels as a numpy matrix (rows: voxels; columns: 1); file *_sigmaval.bin will store voxel-wise noise levels for validation voxels as a numpy matrix (rows: voxels; columns: 1); file *_sigmatest.bin will store voxel-wise noise levels for test voxels as a numpy matrix (rows: voxels; columns: 1)')
	parser.add_argument('--ntrain', metavar='<value>', default='8000', help='number of synthetic voxels for training (default: 8000; note that if multiple SNR levels are requested, the final number of training voxels may change slightly from what has been requested)')
	parser.add_argument('--nval', metavar='<value>', default='2000', help='number of synthetic voxels for validation (default: 2000; note that if multiple SNR levels are requested, the final number of training voxels may change slightly from what has been requested)')	
	parser.add_argument('--ntest', metavar='<value>', default='1000', help='number of synthetic voxels for testing (default: 1000; note that if multiple SNR levels are requested, the final number of training voxels may change slightly from what has been requested)')
	parser.add_argument('--snr', metavar='<value>', default='30', help='value or values (separated by hyphens) of signal-to-noise ratio to be used to corrupt the data with synthetic noise (example: --snr 20 or --snr 20-15-10; default 30; values higher than 1e6 will be mapped to infinity, i.e. no noise added). SNR is evaluated with respect to non-weighted signals (e.g. signal with no inversion pulses, no diffusion-weighting, minimum TE etc)')	
	parser.add_argument('--noise', metavar='<value>', default='gauss', help='synthetic noise type (choose among "gauss" and "rician"; default "gauss")')
	parser.add_argument('--seed', metavar='<value>', default='20180721', help='seed for random number generation (default: 20180721)')	
	parser.add_argument('--bias', metavar='<value>', default='100.0', help='multiplicative constant used to scale all synthetic signals, representing the offset proton density signal level with no weigthings (default: 100.0)')
	parser.add_argument('--parmin', metavar='<value>', help='list of lower bounds of tissue parameters. Entries corresponding to different parameters should be separated by a comma (for example: 0.5,0.2,250,0.5 for model br_sirsmdt). Tissue parameters are: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0; model "twocompdwite", parameters v, Da, t2a, Db, Kb, t2b, s0, where a and b indicate compartments a and b. If not specified, default tissue parameter ranges are used.')
	parser.add_argument('--parmax', metavar='<value>', help='list of upper bounds of tissue parameters. Entries corresponding to different parameters should be separated by a comma (for example: 2.4,0.9,3000,5.0 for model br_sirsmdt). Tissue parameters are: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0; model "twocompdwite", parameters v, Da, t2a, Db, Kb, t2b, s0, where a and b indicate compartments a and b. If not specified, default tissue parameter ranges are used.')	
	args = parser.parse_args()

	### Get input parameters
	outstr = args.out_str
	mrimodel = args.mri_model
	ntrain = int(args.ntrain)
	nval = int(args.nval)
	ntest = int(args.ntest)
	snr = (args.snr).split('-')
	snr = np.array( list(map( float,snr )) )
	snr[snr>1e6] = np.inf
	nsnr = snr.size
	noisetype = args.noise
	myseed = int(args.seed)
	bsig = float(args.bias)

	### Make sure the number of training/test/validation voxels can be divided by the number of SNR levels
	ntrain = int(np.round(float(ntrain)/float(nsnr))*nsnr)
	nval = int(np.round(float(nval)/float(nsnr))*nsnr)
	ntest = int(np.round(float(ntest)/float(nsnr))*nsnr)
	
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
			

	### Print some information
	print('')
	print('')
	print('********************************************************************')
	print('                 SYNTHESISE DATA TO TRAIN A qMRI-NET                ')
	print('********************************************************************')
	print('')
	print('** Output files : {}_*'.format(args.out_str))
	print('** MRI model: {}'.format(args.mri_model))
	print('** MRI protocol file: {}'.format(args.mri_prot))
	print('** Noise type: {}'.format(args.noise))
	print('** SNR levels: {}'.format(snr))
	print('** Seed for random number generators: {}'.format(myseed))
	print('** Unweighted signal bias: {}'.format(bsig))

	### Load MRI protocol
	try:
		mriprot = np.loadtxt(args.mri_prot)
	except: 		
		raise RuntimeError('the MRI protocol is not in a suitable text file format. Sorry!')

	Nmeas = mriprot.shape[1]

	### Check that MRI model exists
	if ( (mrimodel!='pr_hybriddwi') and (mrimodel!='br_sirsmdt') and (mrimodel!='twocompdwite') ):
		raise RuntimeError('the chosen MRI model is not implemented. Sorry!')

	### Check that the noise model exists
	if ( (noisetype!='gauss') and (noisetype!='rician') ):
		raise RuntimeError('the chosen noise model is not implemented. Sorry!')

	### Get a qMRI-net
	if mrimodel=='pr_hybriddwi':
		totpar=9
		s0idx=totpar-1
		qnet = deepqmri.qmrisig([Nmeas,totpar],0.0,'pr_hybriddwi',mriprot)
	elif mrimodel=='br_sirsmdt':
		totpar=4
		s0idx=totpar-1
		qnet = deepqmri.qmrisig([Nmeas,totpar],0.0,'br_sirsmdt',mriprot)
	elif mrimodel=='twocompdwite':
		totpar=7
		s0idx=totpar-1
		qnet = deepqmri.qmrisig([Nmeas,totpar],0.0,'twocompdwite',mriprot)
		
	if (args.parmin is not None) or (args.parmax is not None):
		qnet.changelim(pminbound,pmaxbound)
		
	print('** Tissue parameter lower bounds: {}'.format(qnet.param_min))	
	print('** Tissue parameter upper bounds: {}'.format(qnet.param_max))	

	### Set random seeds
	np.random.seed(myseed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(myseed)    # Random seed for reproducibility: PyTorch

	### Synthesise uniformly-distributed tissue parameters
	print('')
	print('                 ... synthesising tissue parameters')
	npars = qnet.param_min.size
	ptrain = np.zeros((ntrain,npars))
	pval = np.zeros((nval,npars))
	ptest = np.zeros((ntest,npars))

	for pp in range(0, npars):		
		ptrain[:,pp] = qnet.param_min[pp] + (qnet.param_max[pp] - qnet.param_min[pp])*np.random.rand(ntrain) 
		pval[:,pp] = qnet.param_min[pp] + (qnet.param_max[pp] - qnet.param_min[pp])*np.random.rand(nval) 
		ptest[:,pp] = qnet.param_min[pp] + (qnet.param_max[pp] - qnet.param_min[pp])*np.random.rand(ntest) 


	### Predict noise-free signals
	print('')
	print('                 ... generating noise-free signals ({} measurements)'.format(Nmeas))
	strain = qnet.getsignals(Tensor(ptrain))
	strain = strain.detach().numpy()
	strain = bsig*strain

	sval = qnet.getsignals(Tensor(pval))
	sval = sval.detach().numpy()
	sval = bsig*sval

	stest = qnet.getsignals(Tensor(ptest))
	stest = stest.detach().numpy()
	stest = bsig*stest

	### Generate arrays of sigma of noise levels
	print('')
	print('                 ... adding noise')
	snr_train = np.ones((0,1))
	snr_val = np.ones((0,1))
	snr_test = np.ones((0,1))

	for ss in range(0,nsnr):
		snr_train = np.concatenate(  ( snr_train , snr[ss]*np.ones((int(ntrain/nsnr),1)) ) , axis = 0 ) 
		snr_val = np.concatenate(  ( snr_val , snr[ss]*np.ones((int(nval/nsnr),1)) ) , axis = 0 ) 
		snr_test = np.concatenate(  ( snr_test , snr[ss]*np.ones((int(ntest/nsnr),1)) ) , axis = 0 ) 

	np.random.shuffle(snr_train)
	np.random.shuffle(snr_val)
	np.random.shuffle(snr_test)

	sgm_train = (bsig*ptrain[:,s0idx:s0idx+1]) / snr_train
	sgm_val = (bsig*pval[:,s0idx:s0idx+1]) / snr_val
	sgm_test = (bsig*ptest[:,s0idx:s0idx+1]) / snr_test

	### Add noise
	strain_noisy = np.zeros(strain.shape)
	sval_noisy = np.zeros(sval.shape)
	stest_noisy = np.zeros(stest.shape)

	for tt in range(0, ntrain):
	
		if(noisetype=='gauss'):
			strain_noisy[tt,:] = strain[tt,:] + sgm_train[tt]*np.random.randn(1,Nmeas) 		
		
		elif(noisetype=='rician'):
			strain_noisy[tt,:] = np.sqrt( ( strain[tt,:] + sgm_train[tt]*np.random.randn(1,Nmeas) )**2 + ( sgm_train[tt]*np.random.randn(1,Nmeas) )**2  )


	for vv in range(0, nval):
	
		if(noisetype=='gauss'):
			sval_noisy[vv,:] = sval[vv,:] + sgm_val[vv]*np.random.randn(1,Nmeas) 		
		
		elif(noisetype=='rician'):
			sval_noisy[vv,:] = np.sqrt( ( sval[vv,:] + sgm_val[vv]*np.random.randn(1,Nmeas) )**2 + ( sgm_val[vv]*np.random.randn(1,Nmeas) )**2  )

	for qq in range(0, ntest):
	
		if(noisetype=='gauss'):
			stest_noisy[qq,:] = stest[qq,:] + sgm_test[qq]*np.random.randn(1,Nmeas) 		
		
		elif(noisetype=='rician'):
			stest_noisy[qq,:] = np.sqrt( ( stest[qq,:] + sgm_test[qq]*np.random.randn(1,Nmeas) )**2 + ( sgm_test[qq]*np.random.randn(1,Nmeas) )**2  )



	### Save output files
	print('')
	print('                 ... saving output files')
	try:
		my_file = open('{}_sigtrain.bin'.format(outstr),'wb')
		pk.dump(strain_noisy,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigval.bin'.format(outstr),'wb')
		pk.dump(sval_noisy,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigtest.bin'.format(outstr),'wb')
		pk.dump(stest_noisy,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()



		my_file = open('{}_paramtrain.bin'.format(outstr),'wb')
		pk.dump(ptrain,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_paramval.bin'.format(outstr),'wb')
		pk.dump(pval,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_paramtest.bin'.format(outstr),'wb')
		pk.dump(ptest,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()



		my_file = open('{}_sigmatrain.bin'.format(outstr),'wb')
		pk.dump(sgm_train,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigmaval.bin'.format(outstr),'wb')
		pk.dump(sgm_val,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigmatest.bin'.format(outstr),'wb')
		pk.dump(sgm_test,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')

	### Done
	print('')
	print('                 ... done!')
	print('')
	sys.exit(0)


	
