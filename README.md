# qMRINet
The ***quantitative MRI network*** (qMRINet) enables voxel-by-voxel fitting of quantitative MRI models with fully-connected deep neural networks that can be trained on synthetic or actual MRI measurements.

At present qMRINet enables fitting of the following multi-contrast signal models:
* [Hybrid multi-dimensional MRI](http://doi.org/10.1148/radiol.2018171130) for joint diffusion-T2 relaxation imaging of the prostate;
* [T1-weighted diffusion tensors](http://doi.org/10.1016/j.neuroimage.2016.07.037) on [spherical mean](http://doi.org/10.1002/mrm.25734) MRI signals for joint diffusion-T1 relaxation imaging of the brain. 


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
git clone https://github.com/fragrussu/qMRINet.git 
```
4. qMRINet is ready for you in `./qMRINet`. qMRINet tools are available here: 
```
./qMRINet/tools
```
while a number of tutorials are available here:
```
./qMRINet/tutorials
```
5. You should now be able to use the code. Try to print the manual of a script, for instance of `syndata_deepqmri.py`, to make sure this is really the case:
```
python ./qMRINet/tools/syndata_deepqmri.py --help
```

## qMRINet: classes and tutorials
qMRINet is based on 3 different python classes: `qmrisig`, `qmripar` and `qmriinterp`. These are defined in the [deepqmri.py](https://github.com/fragrussu/qMRINet/blob/master/tools/deepqmri.py) file. Specifically:

* `qmrisig` allows you to train a qMRINet by minimising a loss function measuring the mean squared error (MSE, or L2 error norm) *between measured MRI signals and qMRINet signal predictions*;
* `qmripar` allows you to train a qMRINet by minimising a loss function measuring the MSE *between ground truth tissue parameters and qMRINet tissue parameter predictions*;
* `qmriinterp` allows you to learn a resampling between different qMRI protocols (it essentially equivalent to the *predictor* sub-network of a [SARDU-Net](https://github.com/fragrussu/sardunet)).

Additional information can be found in the [tutorial page](https://github.com/fragrussu/qMRINet/blob/master/tutorials/README.md), including: i) details of the signal models you can fit with qMRINet; ii) details on the methods of each class; iii) links to practical tutorials that show qMRINet tools at work.

## Citation
If you use qMRINet, please remember to cite our work:

"*Select and retrieve via direct upsampling* network (SARDU-Net): a data-driven, model-free, deep learning approach for quantitative MRI protocol design". Grussu F, Blumberg SB, Battiston M, Kakkar LS, Lin H, Ianuș A, Schneider T, Singh S, Bourne R, Punwani S, Atkinson D, Gandini Wheeler-Kingshott CAM, Panagiotaki E, Mertzanidou T and Alexander DC. biorxiv 2020, DOI: [10.1101/2020.05.26.116491](https://doi.org/10.1101/2020.05.26.116491). 

## License
qMRINet is distributed under the BSD 2-Clause License, Copyright (c) 2020 University College London. All rights reserved.
Link to license [here](http://github.com/fragrussu/qMRINet/blob/master/LICENSE).

## Acknowledgements
The development of qMRINet was funded by the Engineering and Physical Sciences Research Council (EPSRC EP/R006032/1). This project has also received funding under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541 and No. 666992. Funding has also been received from: EPSRC (M020533/1, G007748, I027084, N018702); Rosetrees Trust (UK, funding FG); Prostate Cancer UK Targeted Call 2014 (Translational Research St.2, project reference PG14-018-TR2); Spinal Research (UK), Wings for Life (Austria), Craig H. Neilsen Foundation (USA) for jointly funding the INSPIRED study; Wings for Life (#169111); UK Multiple Sclerosis Society (grants 892/08 and 77/2017); the Department of Health's National Institute for HealthResearch (NIHR) Biomedical Research Centres and UCLH NIHR Biomedical Research Centre; Champalimaud Centre for the Unknown, Lisbon (Portugal); European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101003390.

