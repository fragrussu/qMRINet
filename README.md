# qMRINet
The ***quantitative MRI network*** (qMRINet) toolbox enables voxel-by-voxel fitting of quantitative MRI (qMRI) models or resampling of qMRI protocols with fully-connected deep neural networks. These can be trained on synthetic or actual MRI measurements.

At present qMRINet enables fitting of the following multi-contrast signal models:
* [Hybrid multi-dimensional MRI](http://doi.org/10.1148/radiol.2018171130) for joint diffusion-T2 relaxation imaging of the prostate;
* [T1-weighted diffusion tensor](http://doi.org/10.1016/j.neuroimage.2016.07.037) modelling of [spherical mean](http://doi.org/10.1002/mrm.25734) signals for joint diffusion-T1 imaging of the brain. 


## Dependencies
To use qMRINet you need a Python 3 distribution such as [Anaconda](http://www.anaconda.com/distribution). Additionally, you need the following third party modules/packages:
* [NumPy](http://numpy.org)
* [Nibabel](http://nipy.org/nibabel)
* [SciPy](http://www.scipy.org)
* [PyTorch](http://pytorch.org/)


## Download 
Getting qMRINet is extremely easy: cloning this repository is all you need to do. The tools would be ready for you to run.

If you use Linux or MacOS:

1. Open a terminal;
2. Navigate to your destination folder;
3. Clone qMRINet:
```
$ git clone https://github.com/fragrussu/qMRINet.git 
```
4. qMRINet is ready for you in `./qMRINet`. qMRINet tools are available here: [`./qMRINet/tools`](https://github.com/fragrussu/qMRINet/tree/master/tools); a user guide and some tutorials are available here: [`./qMRINet/tutorials`](https://github.com/fragrussu/qMRINet/tree/master/tutorials).
5. You should now be able to use the code. Try to print the manual of a tool in your terminal, for instance of `syndata_deepqmri.py`, to make sure this is really the case:
```
$ python ./qMRINet/tools/syndata_deepqmri.py --help
```

## qMRINet: classes and tutorials
qMRINet is based on 3 different python classes: `qmrisig`, `qmripar` and `qmriinterp`. These are defined in the [deepqmri.py](https://github.com/fragrussu/qMRINet/blob/master/tools/deepqmri.py) file. Specifically:

* `qmrisig` allows you to fit signal models with a qMRINet trained by minimising a loss function measuring the mean squared error (MSE, or L2 error norm) *between measured MRI signals and qMRINet signal predictions*;
* `qmripar` allows you to fit signal models with a qMRINet trained by minimising a loss function measuring the MSE *between ground truth tissue parameters and qMRINet tissue parameter predictions*;
* `qmriinterp` allows you to learn a resampling between different qMRI protocols, voxel-by-voxel (it essentially equivalent to the *predictor* sub-network of a [SARDU-Net](https://github.com/fragrussu/sardunet)).

Additional information can be found in this [guide](https://github.com/fragrussu/qMRINet/blob/master/tutorials/README.md), including:
1. details on the methods of each class;
2. details of the signal models you can fit with qMRINet; 
3. practical tutorials that demonstrate how to use qMRINet tools.

## Citation
If you use qMRINet, please remember to cite our work:

"*Select and retrieve via direct upsampling* network (SARDU-Net): a data-driven, model-free, deep learning approach for quantitative MRI protocol design". Grussu F, Blumberg SB, Battiston M, Kakkar LS, Lin H, Ianuș A, Schneider T, Singh S, Bourne R, Punwani S, Atkinson D, Gandini Wheeler-Kingshott CAM, Panagiotaki E, Mertzanidou T and Alexander DC. biorxiv 2020, DOI: [10.1101/2020.05.26.116491](https://doi.org/10.1101/2020.05.26.116491). 

## License
qMRINet is distributed under the BSD 2-Clause License, Copyright (c) 2020 University College London. All rights reserved.
Link to license [here](http://github.com/fragrussu/qMRINet/blob/master/LICENSE).

## Acknowledgements
The development of qMRINet was funded by the Engineering and Physical Sciences Research Council (EPSRC EP/R006032/1). This project has also received funding under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541 and No. 666992. Funding has also been received from: EPSRC (M020533/1, G007748, I027084, N018702); Rosetrees Trust (UK, funding FG); Prostate Cancer UK Targeted Call 2014 (Translational Research St.2, project reference PG14-018-TR2); Spinal Research (UK), Wings for Life (Austria), Craig H. Neilsen Foundation (USA) for jointly funding the INSPIRED study; Wings for Life (#169111); UK Multiple Sclerosis Society (grants 892/08 and 77/2017); the Department of Health's National Institute for HealthResearch (NIHR) Biomedical Research Centres and UCLH NIHR Biomedical Research Centre; Champalimaud Centre for the Unknown, Lisbon (Portugal); European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101003390.

