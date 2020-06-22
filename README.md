# qMRINet
The ***quantitative MRI network*** (qMRINet) toolbox enables voxel-by-voxel fitting of quantitative MRI (qMRI) models and resampling of qMRI protocols with fully-connected deep neural networks.

**>> The code is being tidied up and will be made available soon. Sorry for the delay! <<**

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

## User guide
qMRINet is based on 3 different python classes: `qmrisig`, `qmripar` and `qmriinterp`. These are defined in the [deepqmri.py](https://github.com/fragrussu/qMRINet/blob/master/tools/deepqmri.py) file and described in detail in this [user guide](https://github.com/fragrussu/qMRINet/blob/master/tutorials/README.md). The guide contains:
1. information on qMRINet classes and on their methods, including illustrations;
2. a list of signal models currently implemented in qMRINet; 
3. practical tutorials that demonstrate how to use qMRINet tools.

## Citation
If you use qMRINet, please remember to cite our work:

"*Select and retrieve via direct upsampling* network (SARDU-Net): a data-driven, model-free, deep learning approach for quantitative MRI protocol design". Grussu F, Blumberg SB, Battiston M, Kakkar LS, Lin H, Ianuș A, Schneider T, Singh S, Bourne R, Punwani S, Atkinson D, Gandini Wheeler-Kingshott CAM, Panagiotaki E, Mertzanidou T and Alexander DC. biorxiv 2020, DOI: [10.1101/2020.05.26.116491](https://doi.org/10.1101/2020.05.26.116491). 

## License
qMRINet is distributed under the BSD 2-Clause License, Copyright (c) 2020 University College London. All rights reserved.
Link to license [here](http://github.com/fragrussu/qMRINet/blob/master/LICENSE).

## Acknowledgements
The development of qMRINet was funded by the Engineering and Physical Sciences Research Council (EPSRC EP/R006032/1). This project has also received funding under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541 and No. 666992. Funding has also been received from: EPSRC (M020533/1, G007748, I027084, N018702); Rosetrees Trust (UK, funding FG); Prostate Cancer UK Targeted Call 2014 (Translational Research St.2, project reference PG14-018-TR2); Spinal Research (UK), Wings for Life (Austria), Craig H. Neilsen Foundation (USA) for jointly funding the INSPIRED study; Wings for Life (#169111); UK Multiple Sclerosis Society (grants 892/08 and 77/2017); the Department of Health's National Institute for HealthResearch (NIHR) Biomedical Research Centres and UCLH NIHR Biomedical Research Centre; Champalimaud Centre for the Unknown, Lisbon (Portugal); European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101003390.

