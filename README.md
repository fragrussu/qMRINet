# qMRINet
The ***quantitative MRI network*** (qMRINet) toolbox enables voxel-by-voxel fitting of quantitative MRI (qMRI) models and resampling of qMRI protocols with fully-connected deep neural networks.

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

"Deep learning model fitting for diffusion-relaxometry: a comparative study"; Grussu F, Battiston M, Palombo M, Schneider T, Gandini Wheeler-Kingshott CAM and Alexander DC. Proceedings of the 2020 International MICCAI Workshop on Computational Diffusion MRI (accepted; in production). <ins>Disclosure</ins>: TS worked for Philips (UK) and now works for DeepSpin (Germany).

Earlier preliminary results are available in this preprint: "Deep learning model fitting for diffusion-relaxometry: a comparative study"; Grussu F, Battiston M, Palombo M, Schneider T, Gandini Wheeler-Kingshott CAM and Alexander DC. biorxiv 2020, DOI: [10.1101/2020.10.20.347625](https://doi.org/10.1101/2020.10.20.347625).

## License
qMRINet is distributed under the BSD 2-Clause License, Copyright (c) 2020 University College London. All rights reserved. Link to license [here](http://github.com/fragrussu/qMRINet/blob/master/LICENSE).

## Acknowledgements
The development of qMRINet was funded by the Engineering and Physical Sciences Research Council (EPSRC EP/R006032/1, M020533/1, EP/N018702/1 G007748, I027084) and by the UK Research and Innovation (UKRI) programme (MR/T020296/1, funding M.P.). This project has received funding under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541, and from: Rosetrees Trust (UK, funding F.G.); Spinal Research (UK), Wings for Life (Austria), Craig H. Neilsen Foundation (USA) for INSPIRED; Wings for Life (169111); UK Multiple Sclerosis Society (grants 892/08 and 77/2017); the Department of Health’s National Institute for Health Research (NIHR) Biomedical Research Centres and UCLH NIHR Biomedical Research Centre.


