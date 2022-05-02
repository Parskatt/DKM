# DKM - Deep Kernelized Dense Geometric Matching
Contains code for [*Deep Kernelized Dense Geometric Matching*](https://arxiv.org/abs/2202.00667)

We provide pretrained models and code for evaluation and running on your own images.
We do not curently provide code for training models, but you can basically copy paste the model code into your own training framework and run it.

Note that the performance of current models is greater than in the pre-print.
This is due to continued development since submission.
## Install
Run ``pip install -e .``

## Using a (Pretrained) Model
Models can be imported by:
``` python
from dkm import dkm_base
model = dkm_base(pretrained=True, version="v11")
```
This creates a model, and loads pretrained weights.
## Running on your own images
``` python
from dkm import dkm_base
from PIL import Image
model = dkm_base(pretrained=True, version="v11")
im1, im2 = Image.open("im1.jpg"), Image.open("im2.jpg")
# Note that matches are produced in the normalized grid [-1, 1] x [-1, 1] 
dense_matches, dense_certainty = model.match(im1, im2)
# You may want to process these, e.g. we found dense_certainty = dense_certainty.sqrt() to work quite well in some cases.
# Sample 10000 sparse matches
sparse_matches, sparse_certainty = model.sample(dense_matches, dense_certainty, 10000)
```
## Downloading Benchmarks
### HPatches
First, make sure that the "data/hpatches" path exists. I usually prefer to do this by:

`` ln -s place/where/your/datasets/are/stored/hpatches data/hpatches `` 

Then run (if you don't already have hpatches downloaded)
`` bash scripts/download_hpatches.sh``
### Yfcc100m (OANet Split)
We use the split introduced by OANet, this split can be found from e.g. https://github.com/PruneTruong/DenseMatching
### Megadepth (LoFTR Split)
Currently we do not support the LoFTR split, as we trained on one of the scenes used there.
Future releases may support this split, stay tuned.
### Scannet (SuperGlue Split)
We use the same split of scannet as superglue.
LoFTR provides the split here: https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc

## Evaluation
Here we provide approximate performance numbers for DKM using this codebase.
Note that the randomness involved in geometry estimation means that the numbers are not exact. (+- 0.5 typically)
### HPatches
To evaluate on HPatches Homography Estimation, run:

``` python
from dkm import dkm_base
from dkm.benchmarks import HpatchesHomogBenchmark

model = dkm_base(pretrained=True, version="v11")
homog_benchmark = HpatchesHomogBenchmark("data/hpatches")
homog_benchmark.benchmark_hpatches(model)
```


## Results 

### HPatches Homography Estimation

| | | AUC | |
|-| ----------- | ----------- | --------- |
|| @3px | @5px | @10px |
| LoFTR (CVPR'21) | 65.9 | 75.6 | 84.6 |
| **DKM (Ours)** | **71.2** | **80.6** | **88.7** |


### Scannet Pose Estimation
Here we compare the performance on Scannet of models not trained on Scannet. (For reference we also include the version LoFTR specifically trained on Scannet) 
| | |AUC| | |mAP| |
|-| ----------- | ----------- | --------- | ----------- | ----------- | --------- |
|| @5 | @10 | @20 | @5 | @10 | @20 |
| SuperGlue (CVPR'20) *Trained on Megadepth*| 16.16 | 33.81 | 51.84 | - | - | - |
| LoFTR (CVPR'21) *Trained on Megadepth*| 16.88 | 33.62 | 50.62 | - | - | - |
| LoFTR (CVPR'21) *Trained on Scannet* | *22.06* | *40.8* | *57.62* | - | - | - |
| PDCNet (CVPR'21) *Trained on Megadepth* | 17.70 | 35.02 | 51.75 | 39.93 | 50.17 | 60.87 |
| PDCNet+ (Arxiv) *Trained on Megadepth*|19.02 | 36.90 | 54.25 | 42.93 | 53.13 | 63.95|
| **DKM (Ours)** *Trained on Megadepth*| **22.3** | **42.0** | **60.2** | **48.4** | **59.5** | **70.3** |
| **DKM (Ours)** *Trained on Megadepth* *Square root Confidence Sampling* | **22.9** | **43.6** | **61.4** | **51.2** | **62.1** | **72.0** |



### Yfcc100m Pose Estimation
Here we compare to recent methods using a single forward pass. 
PDC-Net+ using multiple passes comes closer to our method, reaching AUC-5 of 37.51.
However, comparing to that method is somewhat unfair as their inference is much slower.

| | |AUC| | |mAP| |
|-| ----------- | ----------- | --------- | ----------- | ----------- | --------- |
|| @5 | @10 | @20 | @5 | @10 | @20 |
| PDCNet (CVPR'21) | 32.21 | 52.61 | 70.13 | 60.52 | 70.91| 80.30 |
| PDCNet+ (Arxiv) | 34.76 | 55.37 | 72.55 | 63.93 | 73.81 | 82.74 |
| **DKM (Ours)** | **40.0** | **60.2** | **76.2** | **69.8** | **78.5** | **86.1** |


## TODO
- [x] Add Model Code
- [x] Upload Pretrained Models
- [x] Add HPatches Homography Benchmark
- [ ] Add More Benchmarks

## Acknowledgement
We have used code and been inspired by (among others) https://github.com/PruneTruong/DenseMatching , https://github.com/zju3dv/LoFTR , and https://github.com/GrumpyZhou/patch2pix  

## BibTeX
If you find our models useful, please consider citing our paper!
```
@article{edstedt2022deep,
  title={Deep Kernelized Dense Geometric Matching},
  author={Edstedt, Johan and Wadenb{\"a}ck, M{\aa}rten and Felsberg, Michael},
  journal={arXiv preprint arXiv:2202.00667},
  year={2022}
}
```
