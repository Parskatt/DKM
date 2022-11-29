# DKM: Dense Kernelized Feature Matching for Geometry Estimation
Contains code for [*DKM: Dense Kernelized Feature Matching for Geometry Estimation*](https://arxiv.org/abs/2202.00667) 

**Note** The code in this repo is in active development, and the api may change without notice.

⚡ **DKMv3 is out, with improved result on ScanNet1500 (+1.6 AUC@5) and MegaDepth1500 (+3.5 AUC@5)**!
## Benchmark Results

<details>

### Megadepth1500

|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv1 | 54.5  | 70.7 | 82.3 |
| DKMv2 | *56.8*  | *72.3* | *83.2* |
| DKMv3 (paper) | **60.5**  | **74.9** | **85.1** |
| DKMv3 (this repo) | **60.0**  | **74.6** | **84.9** |

### Megadepth 8 Scenes
|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv3 (paper) | **60.5**  | **74.5** | **84.2** |
| DKMv3 (this repo) | **60.4**  | **74.6** | **84.3** |


### ScanNet1500
|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv1 | 24.8  | 44.4 | 61.9 |
| DKMv2 | *28.2*  | *49.2* | *66.6* |
| DKMv3 (paper) | **29.4**  | **50.7** | **68.3** |
| DKMv3 (this repo) | **29.8**  | **50.8** | **68.3** |

</details>

## TODO

<details>

- [x] Initial commit of DKMv3
- [ ] Fix compatability issues between DKM versions
- [ ] St Pauls Cathedral Benchmark
- [ ] Scannet download instructions
- [ ] Update demos for DKMv3

</details>

## Navigating the Code
* Code for models can be found in [dkm/models](dkm/models/)
* Code for benchmarks can be found in [dkm/benchmarks](dkm/benchmarks/)
* Code for reproducing experiments from our paper can be found in [experiments/](experiments/)

## Install
Run ``pip install -e .``

## Demo
**Currently broken!**

A demonstration of our method can be run by:
``` bash
python demo_match.py
```
This runs our model trained on mega on two images I took recently in the wild.

## Benchmarks
See [Benchmarks](docs/benchmarks.md) for details.
## Training
See [Training](docs/training.md) for details.
## Reproducing Results
Given that the required benchmark or training dataset has been downloaded and unpacked, results can be reproduced by running the experiments in the experiments folder.

## Acknowledgements
We have used code and been inspired by https://github.com/PruneTruong/DenseMatching, https://github.com/zju3dv/LoFTR, and https://github.com/GrumpyZhou/patch2pix. We additionally thank the authors of ECO-TR for providing their benchmark.

## BibTeX
If you find our models useful, please consider citing our paper!
```
@article{edstedt2022dkm,
  title={DKM: Dense Kernelized Feature Matching for Geometry Estimation},
  author={Edstedt, Johan and Wadenb{\"a}ck, M{\aa}rten and Athanasiadis, Ioannis and Felsberg, Michael},
  journal={arXiv preprint arXiv:2202.00667},
  year={2022}
}
```
