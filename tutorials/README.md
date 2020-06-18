# qMRINet: user guide
This page contains useful information that you will likely find useful to get started with qMRINet and learn how to fit signal models or resample qMRI protocols. 


## qMRINet classes

qMRINet is based on 3 different python classes: `qmrisig`, `qmripar` and `qmriinterp`, which are defined in the [deepqmri.py](https://github.com/fragrussu/qMRINet/tree/master/tools/deepqmri.py) file. Specifically:


* `qmrisig` allows you to train a qMRINet to fit a signal model by minimising a loss function measuring the mean squared error (MSE, or L2 error norm) *between measured MRI signals and qMRINet signal predictions*. Methods in `qmrisig` objects are:
  * `__init__()`: the constructor, to define the hidden layers and tissue parameter ranges;
  * `getparams()`: to map input qMRI measurements to tissue parameters (it relies on two support methods, `getneurons()` and `getnorm()`); 
  * `getsignals()`: to predict qMRI signals from tissue parameters;
  * `forward()`: to pass data through the entire `qmrisig` network, essentially implementing `forward(M) = getsignals( getparams (M) )`, where `M` are input qMRI measurements. The network is trained by minimising *L = | M - forward(M) |<sup>2</sup>*.
* `qmripar` allows you to train a qMRINet to fit a signal model by minimising a loss function measuring the MSE *between ground truth tissue parameters and qMRINet tissue parameter predictions*. Methods in `qmripar` objects are:
  * `__init__()`: the constructor, to define the hidden layers and tissue parameter ranges;
  * `getparams()`: to map input qMRI measurements to tissue parameters; 
  * `getsignals()`: to predict qMRI signals from tissue parameters;
  * `forward()`: to pass data through the entire `qmripar` network, essentially implementing `forward(M) = getparams(M)`, where `M` are input qMRI measurements. The network is trained by minimising *L = | P - forward(M) |<sup>2</sup>*, where *P* are ground truth tissue parameters.
* `qmriinterp` allows you to train a qMRINet to learn a resampling between two different qMRI protocols. A `qmriinterp` is essentially equivalent to the *predictor* sub-network of a [SARDU-Net](https://github.com/fragrussu/sardunet). Methods in `qmriinterp` objects are:
  * `__init__()`: the constructor, to define the hidden layers;
  * `resample()`: to map an input qMRI protocol to an output qMRI protocol;
  * `forward()`: to pass data through the entire `qmriinterp` network, essentially implementing `forward(Ma) = resample(Ma)`, where `Ma` are qMRI measurements obtained from protocol a). The network is trained by minimising *L = | Mb - forward(Ma) |<sup>2</sup>*, where *Mb* are qMRI measurements from the same voxels but performed with protocol b). Essentially, the trained `qmriinterp` network will learn to map measurements from protocol a) to measurements from protocol b). 


Each class has a detailed *help manual*. From a [Jupyter notebook](https://jupyter.org) or in your python interpreter prompt, you can check the manual by typing something like:
```
>>> import deepqmri
>>> help(deepqmri.qmrisig)
```

## qMRINet command-line tools
qMRINet includes a number of [command-line tools](https://github.com/fragrussu/qMRINet/tree/master/tools) that provide handy interfaces to train and deploy objects from the 3 `qmrisig`, `qmripar` and `qmriinterp` classes.

* `syndata_deepqmri.py` enables synthesis of qMRI measurements from uniformly distributed tisssue parameters, which could be useful for training and testing `qmrisig`, `qmripar` and `qmriinterp` networks;
* `par2sig_deepqmri.py` enables synthesis of qMRI measurements from specif, pre-computed tisssue parameters.
* `extractvoxels_deepqmri.py` enables extraction of voxel measurements from qMRI scans in [NIFTI1 format](https://nifti.nimh.nih.gov/nifti-1), which could be used to train a `qmrinetsig` network;
* `sigsubset_deepqmri.py` enables extraction of subsets of qMRI measurements, which could be used to train a `qmriinterp` network;
* `seqsubset_deepqmri.py` enables extraction of subsets of qMRI sequence parmeters from text files;
* `trainsig_deepqmri.py` enables training of a `qmrinetsig` network;
* `trainpar_deepqmri.py` enables training of a `qmripar` network;
* `traininterp_deepqmri.py` enables training of a `qmriinterp` network;
* `sig2par_qmrisig_deepqmri.py` enables the use of a trained `qmrisig` network to estimate tissue parameters;
* `sig2par_qmripar_deepqmri.py` enables the use of a trained `qmripar` network to estimate tissue parameters;
* `sig2sig_qmriinterp_deepqmri.py` enables the use of a trained `qmriinterp` network to resample qMRI protocols;


## qMRINet signal models

A signal model is identified by a unique string. At present, the following signal models are available:

* [Hybrid multi-dimensional MRI](http://doi.org/10.1148/radiol.2018171130) for joint diffusion-T2 relaxation imaging of the prostate. It is identified by string `'pr_hybriddwi'` and can be used to model prostate qMRI data acquired with multi echo time diffusion-weighted sequences.
* [T1-weighted diffusion tensor](http://doi.org/10.1016/j.neuroimage.2016.07.037) on [spherical mean](http://doi.org/10.1002/mrm.25734) MRI signals for joint diffusion-T1 relaxation imaging of the brain. It is identified by string `'br_sirsmdt'` and can be used to model brain qMRI data acquired with diffusion-weighted inversion recovery or diffusion-weighted saturation inversion recovery sequences.


## qMRINet tutorials
A number of tutorials will be available here soon. Sorry for the wait!


