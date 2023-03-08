# DKM: Dense Kernelized Feature Matching for Geometry Estimation
### [Project Page](https://parskatt.github.io/DKM) | [Paper](https://arxiv.org/abs/2202.00667)
<br/>

> DKM: Dense Kernelized Feature Matching for Geometry Estimation  
> [Johan Edstedt](https://scholar.google.com/citations?user=Ul-vMR0AAAAJ), [Ioannis Athanasiadis](https://scholar.google.com/citations?user=RCAtJgUAAAAJ), [Mårten Wadenbäck](https://scholar.google.com/citations?user=6WRQpCQAAAAJ), [Michael Felsberg](https://scholar.google.com/citations?&user=lkWfR08AAAAJ)  
> CVPR 2023

<video poster="" id="mount_rushmore" autoplay controls muted loop height="100%">
            <source src="assets\mount_rushmore.mp4"
                    type="video/mp4">
          </video>

Contains code for [*DKM: Dense Kernelized Feature Matching for Geometry Estimation*](https://arxiv.org/abs/2202.00667) 

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

## Navigating the Code
* Code for models can be found in [dkm/models](dkm/models/)
* Code for benchmarks can be found in [dkm/benchmarks](dkm/benchmarks/)
* Code for reproducing experiments from our paper can be found in [experiments/](experiments/)

## Install
Run ``pip install -e .``

## Demo

A demonstration of our method can be run by:
``` bash
python demo_match.py
```
This runs our model trained on mega on two images taken from Sacre Coeur.

## Benchmarks
See [Benchmarks](docs/benchmarks.md) for details.
## Training
See [Training](docs/training.md) for details.
## Reproducing Results
Given that the required benchmark or training dataset has been downloaded and unpacked, results can be reproduced by running the experiments in the experiments folder.

## Using DKM matches for estimation
We recommend using the excellent Graph-Cut RANSAC algorithm: https://github.com/danini/graph-cut-ransac

|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv3 (RANSAC) | *60.5*  | *74.9* | *85.1* |
| DKMv3 (GC-RANSAC) | **65.5**  | **78.0** | **86.7** |


## Acknowledgements
We have used code and been inspired by https://github.com/PruneTruong/DenseMatching, https://github.com/zju3dv/LoFTR, and https://github.com/GrumpyZhou/patch2pix. We additionally thank the authors of ECO-TR for providing their benchmark.

## BibTeX
If you find our models useful, please consider citing our paper!
```
@inproceedings{edstedt2023dkm,
title={{DKM}: Dense Kernelized Feature Matching for Geometry Estimation},
author={Edstedt, Johan and Athanasiadis, Ioannis and Wadenbäck, Mårten and Felsberg, Michael},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
year={2023}
}
```