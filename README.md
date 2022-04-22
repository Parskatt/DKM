# DKM - Deep Kernelized Dense Geometric Matching
Contains code for "Deep Kernelized Dense Geometric Matching" (https://arxiv.org/abs/2202.00667)

## Install
Run ``pip install -e .``

## Using a (Pretrained) Model
Models can be imported by:
``` python
from dkm import dkm_base
model = dkm_base(pretrained=True, version="v11")
```
This creates a model, and loads pretrained weights.

## Downloading Benchmarks
### HPatches
First, make sure that the "data/hpatches" path exists. I usually prefer to do this by:

`` ln -s place/where/your/datasets/are/stored/hpatches data/hpatches `` 

Then run (if you don't already have hpatches downloaded)
`` bash scripts/download_hpatches.sh``


## Evaluation
### HPatches
To evaluate on HPatches Homography Estimation, run:

``` python
from dkm import dkm_base
from dkm.benchmarks import HpatchesHomogBenchmark

model = dkm_base(pretrained=True, version="v11")
homog_benchmark = HpatchesHomogBenchmark("data/hpatches")
homog_benchmark.benchmark_hpatches(model)
```

## Running on your own images
``` python
from dkm import dkm_base
from PIL import Image
model = dkm_base(pretrained=True, version="v11")
im1, im2 = Image.open("im1.jpg"), Image.open("im2.jpg")
# Note that matches 
dense_matches = model.match(im1, im2)
sparse_matches, sparse_certainty = model.sample(dense_matches)
```
## Results 

### HPatches Homography Estimation

|| 3px | 5px | 10px |
|-| ----------- | ----------- | --------- |
| LoFTR | 65.9 | 75.6 | 84.6 |
| DKM (Ours) | 71.2 | 80.6 | 88.7 |

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
