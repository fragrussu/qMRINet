# qMRINet: overview
"qMRINet" enables voxel-by-voxel fitting of quantitative MRI models with fully-connected deep neural networks that can be trained on synthetic or actual MRI measurements.

At present qMRINet enables fitting of the following signal models:
* [Hybrid multi-dimensional MRI](http://numpy.org) for joint diffusion-T2 relaxation imaging of the prostate;
* [T1-weighted multi-tensor models](http://doi.org/10.1016/j.neuroimage.2016.07.037) via [powder averaging](http://doi.org/10.1002/mrm.25734 ) for for joint diffusion-T1 relaxation imaging of the brain. 


# Dependencies
To use qMRINet you need a Python 3 distribution such as [Anaconda](http://www.anaconda.com/distribution). Additionally, you need the following third party modules/packages:
* [NumPy](http://numpy.org)
* [Nibabel](http://nipy.org/nibabel)
* [SciPy](http://www.scipy.org)
* [PyTorch](http://pytorch.org/)


# Download 
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
5. You should now be able to use the code. Try to print the manual of a script, for instance of `extractvoxels_sardunet.py`, to make sure this is really the case:
```
python ./qMRINet/tools/syndata_deepqmri.py --help
```

# Citation
If you use qMRINet, please remember to cite our work:

"SARDU-Net: a new method for model-free, data-driven experiment design in quantitative MRI". Grussu F, Blumberg SB, Battiston M, Ianuș A, Singh S, Gong F, Whitaker H, Atkninson D, Gandini Wheeler-Kingshott CAM, Punwani S, Panagiotaki E, Mertzanidou T and Alexander DC. Proceedings of the 2020 virtual annual meeting of the International Society for Magnetic Resonance in Medicine (ISMRM). 

# License
qMRINet is distributed under the BSD 2-Clause License, Copyright (c) 2020 University College London. All rights reserved.
Link to license [here](http://github.com/fragrussu/qMRINet/blob/master/LICENSE).

# Acknowledgements
The development of qMRINet was funded by the Engineering and Physical Sciences Research Council (EPSRC EP/R006032/1). This project has also received funding under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541 and No. 666992. Funding has also been received from: EPSRC (M020533/1, G007748, I027084, N018702); Rosetrees Trust (UK, funding FG); Prostate Cancer UK Targeted Call 2014 (Translational Research St.2, project reference PG14-018-TR2); Spinal Research (UK), Wings for Life (Austria), Craig H. Neilsen Foundation (USA) for jointly funding the INSPIRED study; Wings for Life (#169111); UK Multiple Sclerosis Society (grants 892/08 and 77/2017); the Department of Health's National Institute for HealthResearch (NIHR) Biomedical Research Centres and UCLH NIHR Biomedical Research Centre; Champalimaud Centre for the Unknown, Lisbon (Portugal); European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101003390.

