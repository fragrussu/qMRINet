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
from pathlib import Path as pt
sys.path.insert(0, os.path.dirname(pt(__file__).absolute()) )


if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program extract a subset from a text file storing the sequence parameter values of a multi-contrast quantitative MRI experiment. Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')	
	parser.add_argument('filein', help='path to a text file storing the sequence parameters (entries separated by spaces). Rows: sequence parameters (e.g. TE, TI, b-value, etc); columns: measurements')
	parser.add_argument('fileout', help='path of the output file that will store the sequence parameters corresponding to the measurement subset')
	parser.add_argument('subset', help='path to a text file containing the indices of the measurements to keep in the output (NOTE: indices MUST start from 0 and range from 0 to no. of measurements - 1)')
	args = parser.parse_args()

	### Print some information
	print('')
	print('********************************************************************')
	print('            EXTRACT SEQUENCE PARAMETERS OF A SUB-PROTOCOL          ')
	print('********************************************************************')
	print('')
	print('** Called on file: {}'.format(args.filein))
	print('** Subset file: {}'.format(args.subset))
	print('** Output: {}'.format(args.fileout))
	print('')	

	### Load input data
	print('')
	print('      ... loading data ...')

	## Input sequence parameters
	seqin = np.loadtxt(args.filein)

	## Indices of measurements to keep
	idx = np.loadtxt(args.subset)        # Indices of measurements to keep
	idx = np.array(idx)                  # Convert to array
	idx = idx.astype(int)                # Convert to integer

	## Extract subset
	print('')
	print('      ... extracting subset of sequence parameters ...')
	try:
		seqout = seqin[:,idx]
	except:
		seqout_buff = np.transpose(seqin[idx])
		seqout = np.zeros((1,idx.size))
		for qq in range(0,idx.size):
			seqout[0,qq] = seqout_buff[qq]	

	## Save subset
	print('')
	print('      ... saving output ...')
	np.savetxt(args.fileout, seqout, fmt='%.6f', delimiter=' ')

	## Done
	print('')
	print('     Done!')
	print('')

	sys.exit(0)

