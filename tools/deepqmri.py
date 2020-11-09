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

import sys
import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch import reshape as tshape
from torch import cat as tcat
from torch import exp as texp
from torch import log as tlog
from torch import abs as tabs
from torch import erf as terf
from torch import sqrt as tsqrt
from torch import matmul as tmat


### qMRI net
class qmrisig(nn.Module):
	''' qMRI-Net: neural netork class to perform model fitting in quantitative MRI

	    Author: Francesco Grussu, University College London
			               Queen Square Institute of Neurology
					Centre for Medical Image Computing
		                       <f.grussu@ucl.ac.uk><francegrussu@gmail.com>

	    # Code released under BSD Two-Clause license
	    #
	    # Copyright (c) 2020 University College London. 
	    # All rights reserved.
	    #
	    # Redistribution and use in source and binary forms, with or without modification, are permitted 
	    # provided that the following conditions are met:
	    # 
	    # 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
	    # disclaimer.
	    # 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
	    # disclaimer in the documentation and/or other materials provided with the distribution.
	    # 
	    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
	    # INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
	    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
	    # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
	    # USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
	    # LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
	    # OF THE POSSIBILITY OF SUCH DAMAGE.
	    # 
	    # The views and conclusions contained in the software and documentation are those
	    # of the authors and should not be interpreted as representing official policies,
	    # either expressed or implied, of the FreeBSD Project.
	'''
	# Constructor
	def __init__(self,nneurons,pdrop,mrimodel,mriprot):
		'''Initialise a qMRI-Net to be trained on MRI signals as:
			 
		   mynet = deepqmri.qmrisig(nneurons,pdrop,mrimodel,mriprot)


		   * nneurons:     list or numpy array storing the number of neurons in each layer
                                  (e.g. [<INPUT NEURONS>,<HIDDEN NEURONS LAYER 1>, ... , <OUTPUT NEURONS>],
				         where the number of input neurons equals the number of MRI measurements
				         per voxel and the number of output neurons equal the nuber of tissue parameters
				         to estimate)

		   * pdrop:        dropout probability for each layer of neurons in the net		

		   * mrimodel:     a string identifying the MRI model to fit. Choose among
				   "pr_hybriddwi"  --> for prostate hybrid relaxation-diffusion imaging
				    			  Chatterjee et al, Radiology 2018; 287:864–873
						          http://doi.org/10.1148/radiol.2018171130

				    "br_sirsmdt"   --> for brain saturation inversion recovery diffusion tensor imaging
						        on spherical mean signals
						        De Santis et al, NeuroImage 2016; 141:133-142
							http://doi.org/10.1016/j.neuroimage.2016.07.037
							
				    "twocompdwite"  --> for a simple two-compartment diffuion model without anisotropy
				                        modelling jointly diffusion-T2 dependance. One compartment is
				    			 perfectly Gaussian while the second compartment can include 
				    			 non-Gaussian diffusion (i.e. non-zero kurtosis excess)
				    			    						

		   * mriprot:     MRI protocol stored as a numpy array. The protocol depends on the MRI model used:
				   if mrimodel is "pr_hybriddwi" or "twocompdwite"
                                   |
				     ---> mriprot is a matrix has size 2 x Nmeas, where Nmeas is the number
                                         of MRI measurements in the protocol, which must equal the number of input
                                         neurons (i.e. Nmeas must equal nneurons[0]). mriprot must contain:
					   mriprot[0,:]: b-values in s/mm^2
					   mriprot[1,:]: echo times TE in ms

				   if mrimodel is "br_sirsmdt" 
                                   |
				     ---> mriprot is a matrix has size 3 x Nmeas, where Nmeas is the number
                                         of spherical mean MRI measurements in the protocol, which must 
					  equal the number of input neurons (i.e. must equal nneurons[0]). 
					   mriprot must contain:
					   mriprot[0,:]: preparation time in ms (delay saturation - inversion) 
					   mriprot[1,:]: inversion time in ms (delay inversion - excitation)
					   mriprot[2,:]: b-values in s/mm^2
					   For a pure inversion recovery sequence, use a very long preparation time
					   (i.e. at least 5 times the expected maximum T1)



		'''
		super(qmrisig, self).__init__()
		
		### Store input parameters
		layerlist = []   
		nlayers = np.array(nneurons).shape[0] - 1   # Number of layers
		self.mrimodel = mrimodel                    # MRI signal model
		self.mriprot = mriprot                      # MRI protocol
		self.nneurons = np.array(nneurons)          # Number of hidden neurons in each layer

		### Check that input data are consistent estimate

		# Model pr_hybriddwi
		if self.mrimodel=='pr_hybriddwi':
			if nneurons[-1]!=9:
				raise RuntimeError('the number of output neurons for model "pr_hybriddwi" must be 9')

			if mriprot.shape[0]!=2:
				raise RuntimeError('the size of the numpy matrix storting the MRI protocol is wrong')
		
			self.npars = 9

		# Model br_sirsmdt
		if self.mrimodel=='br_sirsmdt':
			if nneurons[-1]!=4:
				raise RuntimeError('the number of output neurons for model "br_sirsmdt" must be 4')

			if mriprot.shape[0]!=3:
				raise RuntimeError('the size of the numpy matrix storting the MRI protocol is wrong')

			self.npars = 4
			
		# Model twocompdwite
		if self.mrimodel=='twocompdwite':
			if nneurons[-1]!=7:
				raise RuntimeError('the number of output neurons for model "twocompdwite" must be 7')

			if mriprot.shape[0]!=2:
				raise RuntimeError('the size of the numpy matrix storting the MRI protocol is wrong')

			self.npars = 7			
			

		# General consistency of number of input neurons
		if mriprot.shape[1]!=nneurons[0]:
			raise RuntimeError('the number of measurements in the MRI protocol must be equal to the number of input neurons')
					

		### Create layers 

		#   Eeach elementary layer features linearity, ReLu and optional dropout
		for ll in range(nlayers):
			layerlist.append(nn.Linear(nneurons[ll], nneurons[ll+1]))   # Linearity
			layerlist.append(nn.ReLU(True))                             # Non-linearity
			if(ll<nlayers-1):
				layerlist.append(nn.Dropout(p=pdrop))               # Dropout (no dropout for last layer)

		# Add a softplus as last layer to make sure neuronal activations are never 0.0 (min. output: log(2.0)~0.6931
		layerlist.append(nn.Softplus())

		# Store layers
		self.layers = nn.ModuleList(layerlist)

		# Add learnable normalisation factors to convert output neuron activations to tissue parameters 
		normlist = []
		for pp in range(self.npars):
			normlist.append( nn.Linear(1, 1, bias=False) )
		self.sgmnorm = nn.ModuleList(normlist)


		#### Set upper and lower bounds for tissue parameters depending on the signal model

		# Model pr_hybriddwi
		if self.mrimodel=='pr_hybriddwi':
			vl_min = 0.0
			v_min = 0.0
			Dl_min = 2.0
			De_min = 0.1
			Ds_min = 0.7
			t2l_min = 300.0
			t2e_min = 20.0
			t2s_min = 40.0
			s0_min = 0.5

			vl_max = 1.0
			v_max = 1.0
			Dl_max = 3.0
			De_max = 0.7
			Ds_max = 2.0
			t2l_max = 1000.0
			t2e_max = 70.0
			t2s_max = 150.0
			s0_max = 10.0

			self.param_min = np.array([vl_min, v_min, Dl_min, De_min, Ds_min, t2l_min, t2e_min, t2s_min, s0_min]) # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s, S0 (note: ve = v*(1 - vl); vs = 1 - vl - ve)
			self.param_max = np.array([vl_max, v_max, Dl_max, De_max, Ds_max, t2l_max, t2e_max, t2s_max, s0_max]) # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s, S0 (note: ve = v*(1 - vl); vs = 1 - vl - ve)
			self.param_name = ['vl','v','Dl','De','Ds','t2l','t2e','t2s','s0']


		# Model br_sirsmdt
		elif self.mrimodel=='br_sirsmdt':
			dpar_min = 0.01
			kperp_min = 0.0
			t1_min = 100.0
			s0_min = 0.5

			dpar_max = 3.2
			kperp_max = 0.99
			t1_max = 4000.0
			s0_max = 5.0

			self.param_min = np.array([dpar_min, kperp_min, t1_min, s0_min]) # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)
			self.param_max = np.array([dpar_max, kperp_max, t1_max, s0_max]) # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)
			self.param_name = ['dpar', 'kperp', 't1', 's0']
			
			
		# Model twocompdwite	
		elif self.mrimodel=='twocompdwite':
			v_min = 0.0
			da_min = 0.01
			t2a_min = 1.0
			db_min = 0.01
			kb_min = -0.5
			t2b_min = 1.0
			s0_min = 0.5
			
			v_max = 1.0
			da_max = 3.20
			t2a_max = 800.0
			db_max = 3.20
			kb_max = 3.0
			t2b_max = 800.0
			s0_max = 5.0
			
			self.param_min = np.array([v_min,da_min,t2a_min,db_min,kb_min,t2b_min,s0_min]) # Parameter order: v, da, t2a, db, kb, t2b, s0
			self.param_max = np.array([v_max,da_max,t2a_max,db_max,kb_max,t2b_max,s0_max]) # Parameter order: v, da, t2a, db, kb, t2b, s0
			self.param_name = ['v', 'da', 't2a', 'db', 'kb', 't2b', 's0']


	### Calculator of output neuron activations from given MRI measurements
	def getneurons(self, x):
		''' Get output neuron activations from an initialised qMRI-Net

		    u = mynet.getneurons(xin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			      (for a mini-batch, xin has size voxels x measurements)

		    * u:      pytorch Tensor storing the output neuronal activations (if x is a mini-batch of size 
			      voxels x measurements, then u has size voxels x number_of_output_neurons; note that
			      there are as many output neurons as tissue parameters to estimate)  
		'''	

		## Pass MRI signals to layers and get output neuron activations
		for mylayer in self.layers:
			x = mylayer(x)

		## Return neuron activations
		return x


	### Normalise output neurons to convert to tissue parameters
	def getnorm(self, x):
		''' Normalise output neuron activations before taking a sigmoid to convert to tissue parameters

		    uout = mynet.getnorm(uin) 

		    * mynet:  initialised qMRI-Net

		    * uin:    pytorch Tensor storing output neuron activations for one voxels or for a mini-batch 
			      (for a mini-batch, xin has size voxels x measurements)

		    * uout:      pytorch Tensor storing the normalised output neuronal activations (same size as uin)
		'''	

		if x.dim()==1:
			
			# Construct a 1D tensor with one scaling factor per tissue parameter
			normt = Tensor( np.zeros( (self.npars) ) )
			for pp in range(self.npars):
				bt = np.zeros( (self.npars) )
				bt[pp] = 1.0
				bt = Tensor(bt)
				bt = self.sgmnorm[pp](Tensor([1.0]))*bt
				normt = normt + bt

			# Normalise
			normt = tabs(normt)
			x = x*normt
							


		elif x.dim()==2:

			# Construct a tensor with nvoxels x nparameters with scaling factors (all voxels have same scaling factors)
			normt = Tensor( np.zeros( (x.shape[0],self.npars) ) )
			for pp in range(self.npars):			
				bt = np.zeros( (x.shape[0],self.npars) )
				bt[:,pp] = 1.0
				bt = Tensor(bt)
				bt = self.sgmnorm[pp](Tensor([1.0]))*bt
				normt = normt + bt

			# Normalise
			normt = tabs(normt)
			x = x*normt

		else:
			raise RuntimeError('getnorm() processes 1D (signals from one voxels) or 2D (signals from multiple voxels) inputs!') 


		# Return scaled neuron activation
		return x



	### Estimator of tissue parameters from given MRI measurements
	def getparams(self, x):
		''' Get tissue parameters from an initialised qMRI-Net

		    p = mynet.getparams(uin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			      (for a mini-batch, xin has size voxels x measurements)

		    * p:      pytorch Tensor storing the predicted tissue parameters (if x is a mini-batch of size 
			      voxels x measurements, then p has size voxels x number_of_tissue_parameters)  
		'''	

		## Pass MRI signals to layers and get output neuronal activations
		x = self.getneurons(x)   # Note that the last layer is softplus, so x cannot be 0.0

		## Normalise neuronal activations in [0; 1.0] in 4 steps (A, B, C, D):
		x = tlog(x)    # step A - take log as activations can vary over many orders of magnitude
		x = x - tlog(tlog(Tensor([2.0]))) # step B: as the minimum value for x was log(2.0), make sure x is not negative
		x = self.getnorm(x)               # step C: multiply each parameter for a learnable normalisation factor
		x = 2.0*( 1.0 / (1.0 + texp(-x) ) - 0.5 ); # step D: pass through a sigmoid

		## Map normalised neuronal activations to MRI tissue parameter ranges
		
		# Model pr_hybriddwi		
		if self.mrimodel=='pr_hybriddwi':

			# Single voxels
			if x.dim()==1:
				for pp in range(0,self.npars):
					x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]

			# Mini-batch from multiple voxels
			elif x.dim()==2:	

				t_allones = np.ones(( x.shape[0], 1)) 

				max_val = Tensor( np.concatenate( ( self.param_max[0]*t_allones , self.param_max[1]*t_allones , self.param_max[2]*t_allones , self.param_max[3]*t_allones , self.param_max[4]*t_allones , self.param_max[5]*t_allones , self.param_max[6]*t_allones , self.param_max[7]*t_allones , self.param_max[8]*t_allones ) , axis=1 )  )


				min_val = Tensor( np.concatenate( ( self.param_min[0]*t_allones , self.param_min[1]*t_allones , self.param_min[2]*t_allones , self.param_min[3]*t_allones , self.param_min[4]*t_allones , self.param_min[5]*t_allones , self.param_min[6]*t_allones , self.param_min[7]*t_allones , self.param_min[8]*t_allones ) , axis=1 )  )


				x = (max_val - min_val)*x + min_val

		# Model br_sirsmdt
		elif self.mrimodel=='br_sirsmdt':

			# Single voxels
			if x.dim()==1:
				for pp in range(0,self.npars):
					x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]

			# Mini-batch from multiple voxels
			elif x.dim()==2:	

				t_allones = np.ones(( x.shape[0], 1)) 

				max_val = Tensor( np.concatenate( ( self.param_max[0]*t_allones , self.param_max[1]*t_allones , self.param_max[2]*t_allones , self.param_max[3]*t_allones ) , axis=1 )  )


				min_val = Tensor( np.concatenate( ( self.param_min[0]*t_allones , self.param_min[1]*t_allones , self.param_min[2]*t_allones , self.param_min[3]*t_allones ) , axis=1 )  )


				x = (max_val - min_val)*x + min_val
				
							
		# Model twocompdwite		
		elif self.mrimodel=='twocompdwite':

			# Single voxels
			if x.dim()==1:
				for pp in range(0,self.npars):
					x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]

			# Mini-batch from multiple voxels
			elif x.dim()==2:	

				t_allones = np.ones(( x.shape[0], 1)) 

				max_val = Tensor( np.concatenate( ( self.param_max[0]*t_allones , self.param_max[1]*t_allones , self.param_max[2]*t_allones , self.param_max[3]*t_allones , self.param_max[4]*t_allones , self.param_max[5]*t_allones , self.param_max[6]*t_allones ) , axis=1 )  )


				min_val = Tensor( np.concatenate( ( self.param_min[0]*t_allones , self.param_min[1]*t_allones , self.param_min[2]*t_allones , self.param_min[3]*t_allones , self.param_min[4]*t_allones , self.param_min[5]*t_allones , self.param_min[6]*t_allones ) , axis=1 )  )


				x = (max_val - min_val)*x + min_val

		# Return tissue parameters
		return x



	### Computation of MRI signals from tissue parameters
	def getsignals(self,x):
		''' Get MRI signals from tissue parameters using analytical MRI signal models

		    xout = mynet.getsignals(pin) 

		    * mynet:  initialised qMRI-Net

		    * pin:    pytorch Tensor storing the tissue parameters from one voxels or from a mini-batch
			      (if pin is a mini-batch of multiple voxels, it must have size 
			       voxels x number_of_tissue_parameters)

		    * xout:   pytorch Tensor storing the predicted MRI signals given input tissue parameters and
			      according to the analytical model specificed in field mynet.mrimodel  
			      (for a mini-batch of multiple voxels, xout has size voxels x measurements)  
		'''		

		## Get MRI protocol
		mriseq = np.copy(self.mriprot)   # Sequence parameters
		Nmeas = mriseq.shape[1]          # Number of MRI signals to predict (number of measurements in the protocol)

		## Compute MRI signals from input parameter x (microstructural tissue parameters) 

		# Model pr_hybriddwi
		if self.mrimodel=='pr_hybriddwi':  # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s

			bvalues = Tensor(mriseq[0,:]/1000.0)
			tevalues = Tensor(mriseq[1,:])
			
			# Single voxel
			if x.dim()==1:

				# Total MRI signal: lumen + epithelium + stroma
				x = x[8]*x[0]*texp(-bvalues*x[2])*texp(-tevalues/x[5]) + x[8]*(1.0 - x[0])*x[1]*texp(-bvalues*x[3])*texp(-tevalues/x[6]) + x[8]*(1.0 - x[0])*(1.0 - x[1])*texp(-bvalues*x[4])*texp(-tevalues/x[7])

			# Mini-batch of multiple voxels: x has size Nvoxel x Nparams, output must be Nvoxel x Nmeas
			elif x.dim()==2:

				Nvox = x.shape[0]  # Number of voxels
				bvalues = tshape(bvalues, (1,Nmeas)) # b-values
				tevalues = tshape(tevalues, (1,Nmeas)) # echo times

				# Signal from luminal pool
				s_l = tmat( tshape( x[:,8]*x[:,0], (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_l = s_l* texp( tmat(  tshape( x[:,2], (Nvox,1) ) , -bvalues  )  )
				s_l = s_l* texp( tmat(  tshape( 1.0/x[:,5], (Nvox,1) ) , -tevalues  )  )

				# Signal from epithelial pool
				s_e = tmat( tshape( x[:,8]*(1.0 - x[:,0])*x[:,1], (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_e = s_e* texp( tmat(  tshape( x[:,3], (Nvox,1) ) , -bvalues  )  )
				s_e = s_e* texp( tmat(  tshape( 1.0/x[:,6], (Nvox,1) ) , -tevalues  )  )

				# Signal from stromal pool
				s_s = tmat( tshape( x[:,8]*(1.0 - x[:,0])*(1.0 - x[:,1]), (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_s = s_s* texp( tmat(  tshape( x[:,4], (Nvox,1) ) , -bvalues  )  )
				s_s = s_s* texp( tmat(  tshape( 1.0/x[:,7], (Nvox,1) ) , -tevalues  )  )

				# Total MRI signal: lumen + epithelium + stroma
				x = s_l + s_e + s_s


		# Model br_sirsmdt
		elif self.mrimodel=='br_sirsmdt':   # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)

			tdel = Tensor(mriseq[0,:])    # Preparation time
			ti = Tensor(mriseq[1,:])      # Inversion time
			bv = mriseq[2,:]              # b-values
			bv[bv==0] = 0.01              # Replace b = 0 small number --> 0+ to avoid division by zero in spherical mean model
			bv = Tensor(bv/1000.0)        # Convert s/mm^2  to  ms/um^2
			
			sfac = 0.5*np.sqrt(np.pi)
			
			# Single voxel
			if x.dim()==1:
				x = sfac*x[3]*tabs( 1.0 - texp( -ti/x[2] ) - ( 1.0 - texp( -tdel/x[2] ) )*texp( -ti/x[2] )  )*texp(-bv*x[0]*x[1])*terf( tsqrt( bv*(x[0] - x[0]*x[1]) ) )/tsqrt( bv*(x[0] - x[0]*x[1]) )


			# Mini-batch of multiple voxels: x has size Nvoxel x Nparams, output must be Nvoxel x Nmeas
			elif x.dim()==2:

				Nvox = x.shape[0]  # Number of voxels
				bv = tshape(bv, (1,Nmeas))     # b-values
				ti = tshape(ti, (1,Nmeas))     # preparation time
				tdel = tshape(tdel, (1,Nmeas)) # preparation time

				s_tot = sfac*tmat( tshape( x[:,3], (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_tot = s_tot*tabs( 1.0 -   texp( tmat(  tshape( 1.0/x[:,2], (Nvox,1) ) , -ti  )  )  - (1.0 - texp( tmat(  tshape( 1.0/x[:,2], (Nvox,1) ) , -tdel  )  ) )*texp( tmat(  tshape( 1.0/x[:,2], (Nvox,1) ) , -ti  )  )  )
				s_tot = s_tot*texp( tmat(  tshape( x[:,0]*x[:,1], (Nvox,1) ) , -bv  )  )
				s_tot = s_tot*terf( tsqrt(  tmat(  tshape( x[:,0] - x[:,0]*x[:,1], (Nvox,1) ) , bv  )   )  )
				s_tot = s_tot/tsqrt(  tmat(  tshape( x[:,0] - x[:,0]*x[:,1], (Nvox,1) ) , bv  )   )

				x = 1.0*s_tot

		# Model twocompdwite		
		elif self.mrimodel=='twocompdwite':
			
			bvalues = Tensor(mriseq[0,:]/1000.0)
			tevalues = Tensor(mriseq[1,:])			

			# Single voxel
			if x.dim()==1:

				# Total MRI signal: compartment a (gaussian diffusion) + compartment b (non-Gaussian diffusion)
				x = x[6]*x[0]*texp(-bvalues*x[1])*texp(-tevalues/x[2]) + x[6]*(1.0 - x[0])*texp(-bvalues*x[3] + (1.0/6.0)*bvalues*bvalues*x[3]*x[3]*x[4] )*texp(-tevalues/x[5]) 

			# Mini-batch of multiple voxels: x has size Nvoxel x Nparams, output must be Nvoxel x Nmeas
			elif x.dim()==2:
			
				Nvox = x.shape[0]  # Number of voxels
				bvalues = tshape(bvalues, (1,Nmeas)) # b-values
				tevalues = tshape(tevalues, (1,Nmeas)) # echo times

				# Signal from compartment a
				s_a = tmat( tshape( x[:,6]*x[:,0], (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_a = s_a* texp( tmat(  tshape( x[:,1], (Nvox,1) ) , -bvalues  )  )
				s_a = s_a* texp( tmat(  tshape( 1.0/x[:,2], (Nvox,1) ) , -tevalues  )  )

				# Signal from compartment b
				s_b = tmat( tshape( x[:,6]*(1.0 - x[:,0]), (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_b = s_b* texp( tmat(  tshape( x[:,3], (Nvox,1) ) , -bvalues  )  )
				s_b = s_b* texp( tmat(  tshape( x[:,3]*x[:,3]*x[:,4], (Nvox,1) ) , (1.0/6.0)*bvalues*bvalues  )  )
				s_b = s_b* texp( tmat(  tshape( 1.0/x[:,5], (Nvox,1) ) , -tevalues  )  )

				# Total MRI signal: compartment a + compartmet b
				x = s_a + s_b


		# Return predicted MRI signals
		return x
		
		

	### Change limits of tissue parameters    
	def changelim(self, pmin, pmax):
		''' Change limits of tissue parameters
		
		    mynet.changelim(pmin, pmax) 
		
		    * mynet:  initialised qMRI-Net
		 
		    * pmin:   array of new lower bounds for tissue parameters
		              (9 parameters if model is pr_hybriddwi, 
		               4 parameters if model is br_sirsmdt, 
		               7 parameters if model is twocompdwite)
		               
		    * pmax:   array of new upper bounds for tissue parameters
		              (9 parameters if model is pr_hybriddwi, 
		               4 parameters if model is br_sirsmdt, 
		               7 parameters if model is twocompdwite)
		'''	

		# Check number of tissue parameters
		if (len(pmin) != self.npars) or (len(pmax) != self.npars):
			raise RuntimeError('you need to specify bounds for {} tissue parameters with model {}'.format(self.npars,self.mrimodel))


		# Model pr_hybriddwi
		if self.mrimodel=='pr_hybriddwi':
			vl_min = pmin[0]
			v_min = pmin[1]
			Dl_min = pmin[2]
			De_min = pmin[3]
			Ds_min = pmin[4]
			t2l_min = pmin[5]
			t2e_min = pmin[6]
			t2s_min = pmin[7]
			s0_min = pmin[8]

			vl_max = pmax[0]
			v_max = pmax[1]
			Dl_max = pmax[2]
			De_max = pmax[3]
			Ds_max = pmax[4]
			t2l_max = pmax[5]
			t2e_max = pmax[6]
			t2s_max = pmax[7]
			s0_max = pmax[8]

			self.param_min = np.array([vl_min, v_min, Dl_min, De_min, Ds_min, t2l_min, t2e_min, t2s_min, s0_min]) # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s, S0 (note: ve = v*(1 - vl); vs = 1 - vl - ve)
			self.param_max = np.array([vl_max, v_max, Dl_max, De_max, Ds_max, t2l_max, t2e_max, t2s_max, s0_max]) # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s, S0 (note: ve = v*(1 - vl); vs = 1 - vl - ve)

		# Model br_sirsmdt
		elif self.mrimodel=='br_sirsmdt':
			dpar_min = pmin[0]
			kperp_min = pmin[1]
			t1_min = pmin[2]
			s0_min = pmin[3]

			dpar_max = pmax[0]
			kperp_max = pmax[1]
			t1_max = pmax[2]
			s0_max = pmax[3]
			
			self.param_min = np.array([dpar_min, kperp_min, t1_min, s0_min]) # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)
			self.param_max = np.array([dpar_max, kperp_max, t1_max, s0_max]) # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)
			
		# Model twocompdwite
		elif self.mrimodel=='twocompdwite':
			v_min = pmin[0]
			da_min = pmin[1]
			t2a_min = pmin[2]
			db_min = pmin[3]
			kb_min = pmin[4]
			t2b_min = pmin[5]
			s0_min = pmin[6]
			
			v_max = pmax[0]
			da_max = pmax[1]
			t2a_max = pmax[2]
			db_max = pmax[3]
			kb_max = pmax[4]
			t2b_max = pmax[5]
			s0_max = pmax[6]
			
			self.param_min = np.array([v_min,da_min,t2a_min,db_min,kb_min,t2b_min,s0_min]) # Parameter order: v, da, t2a, db, kb, t2b, s0
			self.param_max = np.array([v_max,da_max,t2a_max,db_max,kb_max,t2b_max,s0_max]) # Parameter order: v, da, t2a, db, kb, t2b, s0
			

	### Predictor of MRI signal given a set of MRI measurements    
	def forward(self, x):
		''' Full forward pass of a qMRI-Net, estimating MRI signals given input MRI measurements

		    y = mynet.forward(x) 
		    y = mynet(x) 

		    * mynet:  initialised qMRI-Net

		    * x:      pytorch Tensor storing a set of MRI measurements from one voxel or from a mini-batch
			      (for a mini-batch of multiple voxels, x has size voxels x measurements)

		    * y:      pytorch Tensor storing a prediction of x (it has same size as x)
		'''	

		## Encoder: get prediction of underlying tissue parameters
		x = self.getparams(x)    # From MRI measurements --> to tissue parameters

		## Decoder: predict MRI signal back from the estimated tissue parameters
		x = self.getsignals(x)   # From tissue parameters --> to predicted MRI signals 

		## Return predicted MRI signals given the set of input measurements
		return x










### qMRI net
class qmripar(nn.Module):
	''' qMRI-Net: neural netork class to perform model fitting in quantitative MRI

	    Author: Francesco Grussu, University College London
			               Queen Square Institute of Neurology
					Centre for Medical Image Computing
		                       <f.grussu@ucl.ac.uk><francegrussu@gmail.com>

	    # Code released under BSD Two-Clause license
	    #
	    # Copyright (c) 2020 University College London. 
	    # All rights reserved.
	    #
	    # Redistribution and use in source and binary forms, with or without modification, are permitted 
	    # provided that the following conditions are met:
	    # 
	    # 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
	    # disclaimer.
	    # 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
	    # disclaimer in the documentation and/or other materials provided with the distribution.
	    # 
	    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
	    # INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
	    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
	    # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
	    # USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
	    # LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
	    # OF THE POSSIBILITY OF SUCH DAMAGE.
	    # 
	    # The views and conclusions contained in the software and documentation are those
	    # of the authors and should not be interpreted as representing official policies,
	    # either expressed or implied, of the FreeBSD Project.
	'''
	# Constructor
	def __init__(self,nneurons,pdrop,mrimodel,mriprot):
		'''Initialise a qMRI-Net to be trained on tissue parameters as:
			 
		   mynet = deepqmri.qmripar(nneurons,pdrop,mrimodel,mriprot)


		   * nneurons:     list or numpy array storing the number of neurons in each layer
                                  (e.g. [<INPUT NEURONS>,<HIDDEN NEURONS LAYER 1>, ... , <OUTPUT NEURONS>],
				         where the number of input neurons equals the number of MRI measurements
				         per voxel and the number of output neurons equal the nuber of tissue parameters
				         to estimate)

		   * pdrop:        dropout probability for each layer of neurons in the net		 
  
		   * mrimodel:     a string identifying the MRI model to fit. Choose among
				   "pr_hybriddwi"  --> for prostate hybrid relaxation-diffusion imaging
				    			  Chatterjee et al, Radiology 2018; 287:864–873
						          http://doi.org/10.1148/radiol.2018171130

				    "br_sirsmdt"   --> for brain saturation inversion recovery diffusion tensor imaging
						        on spherical mean signals
						        De Santis et al, NeuroImage 2016; 141:133-142
							http://doi.org/10.1016/j.neuroimage.2016.07.037
							
							
				    "twocompdwite"  --> for a simple two-compartment diffuion model without anisotropy
				                        modelling jointly diffusion-T2 dependance. One compartment is
				    			 perfectly Gaussian while the second compartment can include 
				    			 non-Gaussian diffusion (i.e. non-zero kurtosis excess)
				    			 
		   * mriprot:     MRI protocol stored as a numpy array. The protocol depends on the MRI model used:
				   if mrimodel is "pr_hybriddwi" and "twocompdwite"
                                   |
				     ---> mriprot is a matrix has size 2 x Nmeas, where Nmeas is the number
                                         of MRI measurements in the protocol, which must equal the number of input
                                         neurons (i.e. Nmeas must equal nneurons[0]). mriprot must contain:
					   mriprot[0,:]: b-values in s/mm^2
					   mriprot[1,:]: echo times TE in ms

				   if mrimodel is "br_sirsmdt" 
                                   |
				     ---> mriprot is a matrix has size 3 x Nmeas, where Nmeas is the number
                                         of spherical mean MRI measurements in the protocol, which must 
					  equal the number of input neurons (i.e. must equal nneurons[0]). 
					   mriprot must contain:
					   mriprot[0,:]: preparation time in ms (delay saturation - inversion) 
					   mriprot[1,:]: inversion time in ms (delay inversion - excitation)
					   mriprot[2,:]: b-values in s/mm^2
					   For a pure inversion recovery sequence, use a very long preparation time
					   (i.e. at least 5 times the expected maximum T1)




		'''
		super(qmripar, self).__init__()
		
		### Store input parameters
		layerlist = []   
		nlayers = np.array(nneurons).shape[0] - 1   # Number of layers
		self.mrimodel = mrimodel                    # MRI signal model
		self.mriprot = mriprot                      # MRI protocol
		self.nneurons = np.array(nneurons)          # Number of hidden neurons in each layer

		### Check that input data are consistent estimate

		# Model pr_hybriddwi
		if self.mrimodel=='pr_hybriddwi':
			if nneurons[-1]!=9:
				raise RuntimeError('the number of output neurons for model "pr_hybriddwi" must be 9')

			if mriprot.shape[0]!=2:
				raise RuntimeError('the size of the numpy matrix storting the MRI protocol is wrong')
		
			self.npars = 9

		# Model br_sirsmdt
		if self.mrimodel=='br_sirsmdt':
			if nneurons[-1]!=4:
				raise RuntimeError('the number of output neurons for model "br_sirsmdt" must be 4')

			if mriprot.shape[0]!=3:
				raise RuntimeError('the size of the numpy matrix storting the MRI protocol is wrong')

			self.npars = 4
			
		# Model twocompdwite
		if self.mrimodel=='twocompdwite':
			if nneurons[-1]!=7:
				raise RuntimeError('the number of output neurons for model "twocompdwite" must be 7')

			if mriprot.shape[0]!=2:
				raise RuntimeError('the size of the numpy matrix storting the MRI protocol is wrong')

			self.npars = 7				

		# General consistency of number of input neurons
		if mriprot.shape[1]!=nneurons[0]:
			raise RuntimeError('the number of measurements in the MRI protocol must be equal to the number of input neurons')
			

		### Create layers 

		#   Eeach elementary layer features linearity, ReLu and optional dropout
		for ll in range(nlayers):
			layerlist.append(nn.Linear(nneurons[ll], nneurons[ll+1]))   # Linearity
			layerlist.append(nn.ReLU(True))                             # Non-linearity
			layerlist.append(nn.Dropout(p=pdrop))                       # Dropout

		# Store all layers
		self.layers = nn.ModuleList(layerlist)


		#### Set upper and lower bounds for tissue parameters depending on the signal model

		# Model pr_hybriddwi
		if self.mrimodel=='pr_hybriddwi':
			vl_min = 0.0
			v_min = 0.0
			Dl_min = 2.0
			De_min = 0.1
			Ds_min = 0.7
			t2l_min = 300.0
			t2e_min = 20.0
			t2s_min = 40.0
			s0_min = 0.5

			vl_max = 1.0
			v_max = 1.0
			Dl_max = 3.0
			De_max = 0.7
			Ds_max = 2.0
			t2l_max = 1000.0
			t2e_max = 70.0
			t2s_max = 150.0
			s0_max = 10.0

			self.param_min = np.array([vl_min, v_min, Dl_min, De_min, Ds_min, t2l_min, t2e_min, t2s_min, s0_min]) # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s, S0 (note: ve = v*(1 - vl); vs = 1 - vl - ve)
			self.param_max = np.array([vl_max, v_max, Dl_max, De_max, Ds_max, t2l_max, t2e_max, t2s_max, s0_max]) # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s, S0 (note: ve = v*(1 - vl); vs = 1 - vl - ve)
			self.param_name = ['vl','v','Dl','De','Ds','t2l','t2e','t2s','s0']


		# Model br_sirsmdt
		elif self.mrimodel=='br_sirsmdt':
			dpar_min = 0.01
			kperp_min = 0.0
			t1_min = 100.0
			s0_min = 0.5

			dpar_max = 3.2
			kperp_max = 0.99
			t1_max = 4000.0
			s0_max = 5.0

			self.param_min = np.array([dpar_min, kperp_min, t1_min, s0_min]) # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)
			self.param_max = np.array([dpar_max, kperp_max, t1_max, s0_max]) # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)
			self.param_name = ['dpar', 'kperp', 't1', 's0']
			

		# Model twocompdwite			
		elif self.mrimodel=='twocompdwite':
			v_min = 0.0
			da_min = 0.01
			t2a_min = 1.0
			db_min = 0.01
			kb_min = -0.5
			t2b_min = 1.0
			s0_min = 0.5
			
			v_max = 1.0
			da_max = 3.20
			t2a_max = 800.0
			db_max = 3.20
			kb_max = 3.0
			t2b_max = 800.0
			s0_max = 5.0
			
			self.param_min = np.array([v_min,da_min,t2a_min,db_min,kb_min,t2b_min,s0_min]) # Parameter order: v, da, t2a, db, kb, t2b, s0
			self.param_max = np.array([v_max,da_max,t2a_max,db_max,kb_max,t2b_max,s0_max]) # Parameter order: v, da, t2a, db, kb, t2b, s0
			self.param_name = ['v', 'da', 't2a', 'db', 'kb', 't2b', 's0']



	### Calculator of output neuron activations from given MRI measurements
	def getparams(self, x):
		''' Get output neuron activations from an initialised qMRI-Net

		    u = mynet.getneurons(xin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			      (for a mini-batch, xin has size voxels x measurements)

		    * u:      pytorch Tensor storing the output neuronal activations, which during training will
			      be mapped to tissue parameters (if xin is a mini-batch of size 
			      voxels x measurements, then u has size voxels x number_of_output_neurons; note that
			      there are as many output neurons as tissue parameters to estimate)  
		'''	

		## Pass MRI signals to layers and get output neuron activations
		for mylayer in self.layers:
			x = mylayer(x)

		## Return neuron activations
		return x



	### Computation of MRI signals from tissue parameters
	def getsignals(self,x):
		''' Get MRI signals from tissue parameters using analytical MRI signal models

		    xout = mynet.getsignals(pin) 

		    * mynet:  initialised qMRI-Net

		    * pin:    pytorch Tensor storing the tissue parameters from one voxels or from a mini-batch
			      (if pin is a mini-batch of multiple voxels, it must have size 
			       voxels x number_of_tissue_parameters)

		    * xout:   pytorch Tensor storing the predicted MRI signals given input tissue parameters and
			      according to the analytical model specificed in field mynet.mrimodel  
			      (for a mini-batch of multiple voxels, xout has size voxels x measurements)  
		'''		

		## Get MRI protocol
		mriseq = np.copy(self.mriprot)   # Sequence parameters
		Nmeas = mriseq.shape[1] # Number of MRI signals to predict (number of measurements in the protocol)

		## Compute MRI signals from input parameter x (microstructural tissue parameters) 

		# Model pr_hybriddwi
		if self.mrimodel=='pr_hybriddwi':  # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s

			bvalues = Tensor(mriseq[0,:]/1000.0)
			tevalues = Tensor(mriseq[1,:])
			
			# Single voxel
			if x.dim()==1:

				# Total MRI signal: lumen + epithelium + stroma
				x = x[8]*x[0]*texp(-bvalues*x[2])*texp(-tevalues/x[5]) + x[8]*(1.0 - x[0])*x[1]*texp(-bvalues*x[3])*texp(-tevalues/x[6]) + x[8]*(1.0 - x[0])*(1.0 - x[1])*texp(-bvalues*x[4])*texp(-tevalues/x[7])

			# Mini-batch of multiple voxels: x has size Nvoxel x Nparams, output must be Nvoxel x Nmeas
			elif x.dim()==2:

				Nvox = x.shape[0]  # Number of voxels
				bvalues = tshape(bvalues, (1,Nmeas)) # b-values
				tevalues = tshape(tevalues, (1,Nmeas)) # echo times

				# Signal from luminal pool
				s_l = tmat( tshape( x[:,8]*x[:,0], (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_l = s_l* texp( tmat(  tshape( x[:,2], (Nvox,1) ) , -bvalues  )  )
				s_l = s_l* texp( tmat(  tshape( 1.0/x[:,5], (Nvox,1) ) , -tevalues  )  )

				# Signal from epithelial pool
				s_e = tmat( tshape( x[:,8]*(1.0 - x[:,0])*x[:,1], (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_e = s_e* texp( tmat(  tshape( x[:,3], (Nvox,1) ) , -bvalues  )  )
				s_e = s_e* texp( tmat(  tshape( 1.0/x[:,6], (Nvox,1) ) , -tevalues  )  )

				# Signal from stromal pool
				s_s = tmat( tshape( x[:,8]*(1.0 - x[:,0])*(1.0 - x[:,1]), (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_s = s_s* texp( tmat(  tshape( x[:,4], (Nvox,1) ) , -bvalues  )  )
				s_s = s_s* texp( tmat(  tshape( 1.0/x[:,7], (Nvox,1) ) , -tevalues  )  )

				# Total MRI signal: lumen + epithelium + stroma
				x = s_l + s_e + s_s


		# Model br_sirsmdt
		elif self.mrimodel=='br_sirsmdt':   # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)

			tdel = Tensor(mriseq[0,:])    # Preparation time
			ti = Tensor(mriseq[1,:])      # Inversion time
			bv = mriseq[2,:]              # b-values
			bv[bv==0] = 0.01              # Replace b = 0 small number --> 0+ to avoid division by zero in spherical mean model
			bv = Tensor(bv/1000.0)        # Convert s/mm^2  to  ms/um^2
			
			sfac = 0.5*np.sqrt(np.pi)
			
			# Single voxel
			if x.dim()==1:
				x = sfac*x[3]*tabs( 1.0 - texp( -ti/x[2] ) - ( 1.0 - texp( -tdel/x[2] ) )*texp( -ti/x[2] )  )*texp(-bv*x[0]*x[1])*terf( tsqrt( bv*(x[0] - x[0]*x[1]) ) )/tsqrt( bv*(x[0] - x[0]*x[1]) )


			# Mini-batch of multiple voxels: x has size Nvoxel x Nparams, output must be Nvoxel x Nmeas
			elif x.dim()==2:

				Nvox = x.shape[0]  # Number of voxels
				bv = tshape(bv, (1,Nmeas))     # b-values
				ti = tshape(ti, (1,Nmeas))     # preparation time
				tdel = tshape(tdel, (1,Nmeas)) # preparation time

				s_tot = sfac*tmat( tshape( x[:,3], (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_tot = s_tot*tabs( 1.0 -   texp( tmat(  tshape( 1.0/x[:,2], (Nvox,1) ) , -ti  )  )  - (1.0 - texp( tmat(  tshape( 1.0/x[:,2], (Nvox,1) ) , -tdel  )  ) )*texp( tmat(  tshape( 1.0/x[:,2], (Nvox,1) ) , -ti  )  )  )
				s_tot = s_tot*texp( tmat(  tshape( x[:,0]*x[:,1], (Nvox,1) ) , -bv  )  )
				s_tot = s_tot*terf( tsqrt(  tmat(  tshape( x[:,0] - x[:,0]*x[:,1], (Nvox,1) ) , bv  )   )  )
				s_tot = s_tot/tsqrt(  tmat(  tshape( x[:,0] - x[:,0]*x[:,1], (Nvox,1) ) , bv  )   )

				x = 1.0*s_tot
				
				
		# Model twocompdwite		
		elif self.mrimodel=='twocompdwite':
			
			bvalues = Tensor(mriseq[0,:]/1000.0)
			tevalues = Tensor(mriseq[1,:])			

			# Single voxel
			if x.dim()==1:

				# Total MRI signal: compartment a (gaussian diffusion) + compartment b (non-Gaussian diffusion)
				x = x[6]*x[0]*texp(-bvalues*x[1])*texp(-tevalues/x[2]) + x[6]*(1.0 - x[0])*texp(-bvalues*x[3] + (1.0/6.0)*bvalues*bvalues*x[3]*x[3]*x[4] )*texp(-tevalues/x[5]) 

			# Mini-batch of multiple voxels: x has size Nvoxel x Nparams, output must be Nvoxel x Nmeas
			elif x.dim()==2:
			
				Nvox = x.shape[0]  # Number of voxels
				bvalues = tshape(bvalues, (1,Nmeas)) # b-values
				tevalues = tshape(tevalues, (1,Nmeas)) # echo times

				# Signal from compartment a
				s_a = tmat( tshape( x[:,6]*x[:,0], (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_a = s_a* texp( tmat(  tshape( x[:,1], (Nvox,1) ) , -bvalues  )  )
				s_a = s_a* texp( tmat(  tshape( 1.0/x[:,2], (Nvox,1) ) , -tevalues  )  )

				# Signal from compartment b
				s_b = tmat( tshape( x[:,6]*(1.0 - x[:,0]), (Nvox,1) ) , Tensor(np.ones((1,Nmeas))) )
				s_b = s_b* texp( tmat(  tshape( x[:,3], (Nvox,1) ) , -bvalues  )  )
				s_b = s_b* texp( tmat(  tshape( x[:,3]*x[:,3]*x[:,4], (Nvox,1) ) , (1.0/6.0)*bvalues*bvalues  )  )
				s_b = s_b* texp( tmat(  tshape( 1.0/x[:,5], (Nvox,1) ) , -tevalues  )  )

				# Total MRI signal: compartment a + compartmet b
				x = s_a + s_b
				

		# Return predicted MRI signals
		return x



	### Change limits of tissue parameters    
	def changelim(self, pmin, pmax):
		'''Change limits of tissue parameters
		
		    mynet.changelim(pmin, pmax) 
		
		    * mynet:  initialised qMRI-Net
		 
		    * pmin:   array of new lower bounds for tissue parameters
		              (9 parameters if model is pr_hybriddwi, 
		               4 parameters if model is br_sirsmdt, 
		               7 parameters if model is twocompdwite)
		               
		    * pmax:   array of new upper bounds for tissue parameters
		              (9 parameters if model is pr_hybriddwi, 
		               4 parameters if model is br_sirsmdt, 
		               7 parameters if model is twocompdwite)
		'''	

		# Check number of tissue parameters
		if (len(pmin) != self.npars) or (len(pmax) != self.npars):
			raise RuntimeError('you need to specify bounds for {} tissue parameters with model {}'.format(self.npars,self.mrimodel))


		# Model pr_hybriddwi
		if self.mrimodel=='pr_hybriddwi':
			vl_min = pmin[0]
			v_min = pmin[1]
			Dl_min = pmin[2]
			De_min = pmin[3]
			Ds_min = pmin[4]
			t2l_min = pmin[5]
			t2e_min = pmin[6]
			t2s_min = pmin[7]
			s0_min = pmin[8]

			vl_max = pmax[0]
			v_max = pmax[1]
			Dl_max = pmax[2]
			De_max = pmax[3]
			Ds_max = pmax[4]
			t2l_max = pmax[5]
			t2e_max = pmax[6]
			t2s_max = pmax[7]
			s0_max = pmax[8]

			self.param_min = np.array([vl_min, v_min, Dl_min, De_min, Ds_min, t2l_min, t2e_min, t2s_min, s0_min]) # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s, S0 (note: ve = v*(1 - vl); vs = 1 - vl - ve)
			self.param_max = np.array([vl_max, v_max, Dl_max, De_max, Ds_max, t2l_max, t2e_max, t2s_max, s0_max]) # Parameter order: vl, v, Dl, De, Ds, t2l, t2e, t2s, S0 (note: ve = v*(1 - vl); vs = 1 - vl - ve)

		# Model br_sirsmdt
		elif self.mrimodel=='br_sirsmdt':
			dpar_min = pmin[0]
			kperp_min = pmin[1]
			t1_min = pmin[2]
			s0_min = pmin[3]

			dpar_max = pmax[0]
			kperp_max = pmax[1]
			t1_max = pmax[2]
			s0_max = pmax[3]
			
			self.param_min = np.array([dpar_min, kperp_min, t1_min, s0_min]) # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)
			self.param_max = np.array([dpar_max, kperp_max, t1_max, s0_max]) # Parameter order: dpar, kperp, t1, s0 (note: dperp = kperp*dpar)
			
		# Model twocompdwite
		elif self.mrimodel=='twocompdwite':
			v_min = pmin[0]
			da_min = pmin[1]
			t2a_min = pmin[2]
			db_min = pmin[3]
			kb_min = pmin[4]
			t2b_min = pmin[5]
			s0_min = pmin[6]
			
			v_max = pmax[0]
			da_max = pmax[1]
			t2a_max = pmax[2]
			db_max = pmax[3]
			kb_max = pmax[4]
			t2b_max = pmax[5]
			s0_max = pmax[6]
			
			self.param_min = np.array([v_min,da_min,t2a_min,db_min,kb_min,t2b_min,s0_min]) # Parameter order: v, da, t2a, db, kb, t2b, s0
			self.param_max = np.array([v_max,da_max,t2a_max,db_max,kb_max,t2b_max,s0_max]) # Parameter order: v, da, t2a, db, kb, t2b, s0
			


	### Predictor of MRI signal given a set of MRI measurements    
	def forward(self, x):
		''' Full forward pass of a qMRI-Net, estimating MRI signals given input MRI measurements

		    y = mynet.forward(x) 
		    y = mynet(x) 

		    * mynet:  initialised qMRI-Net

		    * x:      pytorch Tensor storing a set of MRI measurements from one voxel or from a mini-batch
			      (for a mini-batch of multiple voxels, x has size voxels x measurements)

		    * y:      pytorch Tensor storing a prediction of x (it has same size as x)
		'''	

		## Get prediction of tissue parameters from MRI measurements
		x = self.getparams(x)    

		## Return tissue parameters
		return x






### qMRI net
class qmriinterp(nn.Module):
	''' qMRI-Net: neural netork class to resample quantitative MRI experiments

	    Author: Francesco Grussu, University College London
			               Queen Square Institute of Neurology
					Centre for Medical Image Computing
		                       <f.grussu@ucl.ac.uk><francegrussu@gmail.com>

	    # Code released under BSD Two-Clause license
	    #
	    # Copyright (c) 2020 University College London. 
	    # All rights reserved.
	    #
	    # Redistribution and use in source and binary forms, with or without modification, are permitted 
	    # provided that the following conditions are met:
	    # 
	    # 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
	    # disclaimer.
	    # 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
	    # disclaimer in the documentation and/or other materials provided with the distribution.
	    # 
	    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
	    # INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
	    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
	    # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
	    # USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
	    # LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
	    # OF THE POSSIBILITY OF SUCH DAMAGE.
	    # 
	    # The views and conclusions contained in the software and documentation are those
	    # of the authors and should not be interpreted as representing official policies,
	    # either expressed or implied, of the FreeBSD Project.
	'''
	# Constructor
	def __init__(self,nneurons,pdrop,act):
		'''Initialise a qMRI-Net to be trained on MRI signals from the same voxel but acquired with different protocols/scanners:
			 
		   mynet = deepqmri.qmriinterp(nneurons,pdrop,act)


		   * nneurons:     list or numpy array storing the number of neurons in each layer
                                  (e.g. [<INPUT NEURONS>,<HIDDEN NEURONS LAYER 1>, ... , <OUTPUT NEURONS>],
				         where the number of input neurons equals the number of input MRI measurements
				         per voxel and the number of output neurons equal the nuber of output MRI 
					 measurements)

		   * pdrop:        dropout probability for each layer of neurons in the net	

		   * act:          string selecting the type of activation for hidden neurons. Choose among:
                                  "ReLU", "LeakyReLU", "RReLU", "Sigmoid"

		'''
		super(qmriinterp, self).__init__()
		
		### Store input parameters
		layerlist = []   
		nlayers = np.array(nneurons).shape[0] - 1   # Number of layers
		self.nneurons = np.array(nneurons)          # Number of hidden neurons in each layer

		### Create layers 

		#   Eeach elementary layer features linearity, ReLu and optional dropout (last layer: only linearity)
		for ll in range(nlayers):
			layerlist.append(nn.Linear(nneurons[ll], nneurons[ll+1]))   # Linearity
			if ll < nlayers - 1:

				# Non-linearity
				if act=="ReLU":
					layerlist.append(nn.ReLU(True))
				elif act=="LeakyReLU":
					layerlist.append(nn.LeakyReLU(inplace=True))
				elif act=="RReLU":
					layerlist.append(nn.RReLU(inplace=True))
				elif act=="Sigmoid":
					layerlist.append(nn.Sigmoid())
				else:
					raise RuntimeError('Non-linear layer not supported!')

				layerlist.append(nn.Dropout(p=pdrop))                       # Dropout

		# Store all layers
		self.layers = nn.ModuleList(layerlist)



	### Calculator of output neuron activations from given MRI measurements
	def resample(self, x):
		''' Get output neuron activations from an initialised qMRI-Net

		    xout = mynet.resample(xin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			      (for a mini-batch, xin has size voxels x measurements)

		    * xout:   pytorch Tensor storing the MRI measurements from the same voxels resampled to the
			      new MRI protocol (if xin is a mini-batch of size voxels x number_of_input_measurements, 
			      then xout has size voxels x number_of_output_measurements)  
		'''	

		## Pass MRI signals to layers and get output neuron activations
		for mylayer in self.layers:
			x = mylayer(x)

		## Return neuron activations
		return x



	### Predictor of MRI signal given a set of MRI measurements    
	def forward(self, x):
		''' Full forward pass of a qMRI-Net, estimating MRI signals given input MRI measurements

		    y = mynet.forward(x) 
		    y = mynet(x) 

		    * mynet:  initialised qMRI-Net

		    * x:      pytorch Tensor storing a set of MRI measurements from one voxel or from a mini-batch
			      (for a mini-batch of multiple voxels, x has size voxels x measurements), acquired 
			       according to an input MR protocol

		    * y:      pytorch Tensor storing the output MRI measurements from the same voxels as x, resampled to
			      a new output MRI protocol
		'''	

		## Get prediction of MRI measurements resampled to a new, output MRI protocol
		x = self.resample(x)    

		## Return resampled MRI measurements
		return x

				
