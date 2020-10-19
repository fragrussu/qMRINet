# qMRINet user guide
This page contains useful information on qMRINet. 


## qMRINet classes

qMRINet is based on 3 different python classes: `qmrisig`, `qmripar` and `qmriinterp`, which are defined in the [deepqmri.py](https://github.com/fragrussu/qMRINet/tree/master/tools/deepqmri.py) file. These are:


* `qmrisig` allows you to train a qMRINet to fit a signal model by minimising a loss function measuring the mean squared error (MSE, or L2 error norm) *between measured MRI signals and qMRINet signal predictions*. Methods in `qmrisig` objects are:
  * `__init__()`: the constructor, to define the hidden layers and tissue parameter ranges;
  * `getparams()`: to map input qMRI measurements to tissue parameters. It relies on two additional methods that implement intermediate normalisation steps:
    * `getneurons()`: to calculate output neuron activations; 
    * `getnorm()`: to normalise output neuron activations so that they can be mapped to tissue parameters; 
  * `getsignals()`: to predict qMRI signals from tissue parameters;
  * `changelim()`: to change the range of tissue parameters from the default settings;
  * `forward()`: to pass data through the entire `qmrisig` network, essentially implementing `forward(s) = getsignals( getparams (s) )`, where `s` are input qMRI measurements. The network is trained by minimising the loss function *L = | s - forward(s) |<sup>2</sup>*, as shown in the figure below.
<img src="https://github.com/fragrussu/qMRINet/blob/master/tutorials/qmrisig.png" width="668"> 

* `qmripar` allows you to train a qMRINet to fit a signal model by minimising a loss function measuring the MSE *between ground truth tissue parameters and qMRINet tissue parameter predictions*. Methods in `qmripar` objects are:
  * `__init__()`: the constructor, to define the hidden layers and tissue parameter ranges;
  * `getparams()`: to map input qMRI measurements to tissue parameters; 
  * `getsignals()`: to predict qMRI signals from tissue parameters;
  * `changelim()`: to change the range of tissue parameters from the default settings;
  * `forward()`: to pass data through the entire `qmripar` network, essentially implementing `forward(s) = getparams(s)`, where `s` are input qMRI measurements. The network is trained by minimising the loss function *L = | p - forward(s) |<sup>2</sup>*, where *p* are ground truth tissue parameters, as illustrated in the figure below.
<img src="https://github.com/fragrussu/qMRINet/blob/master/tutorials/qmripar.png" width="500">

* `qmriinterp` allows you to train a qMRINet to learn a resampling between two different qMRI protocols. A `qmriinterp` network is essentially equivalent to the *predictor* sub-network of a [SARDU-Net](https://github.com/fragrussu/sardunet). Methods in `qmriinterp` objects are:
  * `__init__()`: the constructor, to define the hidden layers;
  * `resample()`: to map an input qMRI protocol to an output qMRI protocol;
  * `forward()`: to pass data through the entire `qmriinterp` network, essentially implementing `forward(sA) = resample(sA)`, where `sA` are qMRI measurements obtained from protocol A. The network is trained by minimising the loss function *L = | sB - forward(sA) |<sup>2</sup>*, where *sB* are qMRI measurements from the same voxels but performed with protocol B. Essentially, the trained `qmriinterp` network will learn to map measurements from protocol A to measurements from protocol B, as illustrated in the figure below.
<img src="https://github.com/fragrussu/qMRINet/blob/master/tutorials/qmriinterp.png" width="500">


Each class has a detailed *help manual*. From a [Jupyter notebook](https://jupyter.org) or in your python interpreter prompt, you can check the manual by typing something like:
```
>>> import deepqmri
>>> help(deepqmri.qmrisig)
```
to print the manual (in this example of the `qmrisig` class).

## qMRINet command-line tools
qMRINet includes a number of [command-line tools](https://github.com/fragrussu/qMRINet/tree/master/tools) that provide handy interfaces to train and deploy objects from the 3 `qmrisig`, `qmripar` and `qmriinterp` classes.

* [`syndata_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/syndata_deepqmri.py) enables synthesis of qMRI measurements from uniformly distributed tisssue parameters, which could be useful for training and testing `qmrisig`, `qmripar` and `qmriinterp` networks;
* [`par2sig_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/par2sig_deepqmri.py) enables synthesis of qMRI measurements from specif, pre-computed tisssue parameters.
* [`extractvoxels_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/extractvoxels_deepqmri.py) enables extraction of voxel measurements from qMRI scans in [NIFTI1 format](https://nifti.nimh.nih.gov/nifti-1), which could be used to train a `qmrinetsig` network;
* [`sigsubset_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/sigsubset_deepqmri.py) enables extraction of subsets of qMRI measurements, which could be used to train a `qmriinterp` network;
* [`seqsubset_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/seqsubset_deepqmri.py) enables extraction of subsets of qMRI sequence parmeters from text files;
* [`trainsig_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/trainsig_deepqmri.py) enables training of a `qmrinetsig` network (at present only on CPU);
* [`trainpar_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/trainpar_deepqmri.py) enables training of a `qmripar` network (at present only on CPU);
* [`traininterp_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/traininterp_deepqmri.py) enables training of a `qmriinterp` network (at present only on CPU);
* [`sig2par_qmrisig_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/sig2par_qmrisig_deepqmri.py) enables the use of a trained `qmrisig` network to estimate tissue parameters;
* [`sig2par_qmripar_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/sig2par_qmripar_deepqmri.py) enables the use of a trained `qmripar` network to estimate tissue parameters;
* [`sig2sig_qmriinterp_deepqmri.py`](https://github.com/fragrussu/qMRINet/blob/master/tools/sig2sig_qmriinterp_deepqmri.py) enables the use of a trained `qmriinterp` network to resample qMRI protocols;

All tools listed above also have a detailed *help manual*, which you can print by simply typing in your terminal `python </PATH/TO/TOOL> --help` (e.g. `python trainsig_deepqmri.py --help`). 

## qMRINet signal models

A signal model is identified by a unique string. At present, the following signal models are available:

* [Hybrid multi-dimensional MRI](http://doi.org/10.1148/radiol.2018171130) for prostate diffusion-T2 relaxation imaging.
  * It is identified by string `'pr_hybriddwi'` and can be used to model prostate qMRI data acquired with multi echo time diffusion-weighted sequences.
  * **Sequence parameters are specified using text files** containing a matrix where the 1st row stores *b*-values in s/mm<sup>2</sup>, while 2nd row echo times in ms (use spaces to separate different entries). Below you find the signal model, sequence parameters and tissue parameters in the same order as outputted by fitting routines.
<img src="https://github.com/fragrussu/qMRINet/blob/master/tutorials/modelpr.png" width="612">
  
* [T1-weighted diffusion tensor](http://doi.org/10.1016/j.neuroimage.2016.07.037) modelling of [directionally-averaged](http://doi.org/10.1002/mrm.25734) signals for brain diffusion-T1 relaxation imaging.
  * It is identified by string `'br_sirsmdt'` and can be used to model brain qMRI data acquired with diffusion-weighted inversion recovery or diffusion-weighted saturation inversion recovery sequences. 
  * **Sequence parameters are specified using text files** containing a matrix where the 1st row stores preparation times (saturation-inversion delay) in ms, the 2nd row inversion times (inversion-excitation delay) in ms, the 3rd row *b*-values in s/mm<sup>2</sup>. For a pure inversion recovery (i.e. no saturation pulse), use a very large number for the saturation-inversion delay (at least 5 times the maximum expected T1); use spaces to separate different entries. Below you find the signal model, sequence parameters and tissue parameters in the same order as outputted by fitting routines.
<img src="https://github.com/fragrussu/qMRINet/blob/master/tutorials/modelbr.png" width="812">


## qMRINet tutorials
A number of tutorials will be available here soon.
