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

## Evaluation
To evaluate on HPatches Homography Estimation, run:

``` python
from dkm import dkm_base
from dkm.benchmarks import HpatchesHomogBenchmark

model = dkm_base(pretrained=True, version="v11")
homog_benchmark = HpatchesHomogBenchmark("data/hpatches")
homog_benchmark.benchmark_hpatches(model)
```


## TODO
- [x] Add Model Code
- [x] Upload Pretrained Models
- [x] Add HPatches Homography Benchmark
- [ ] Add More Benchmarks
