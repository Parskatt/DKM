# DKM - Deep Kernelized Dense Geometric Matching
Contains code for [*Deep Kernelized Dense Geometric Matching*](https://arxiv.org/abs/2202.00667) 

**Note** The code in this repo is in active development, and the api may change without notice.

⚡ **DKMv2 is out, with improved result on ScanNet1500 and MegaDepth1500**!
### Megadepth1500
|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv1 | 54.5  | 70.7 | 82.3 |
| DKMv2 | **56.8**  | **72.3** | **83.2** |

### ScanNet1500
|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv1 | 24.8  | 44.4 | 61.9 |
| DKMv2 | **28.2**  | **49.2** | **66.6** |

## TODO
- [ ] Provide updated training and higher resolution options for DKMv2

## Navigating the Code
* Code for models can be found in dkm/models
* Code for benchmarks can be found in dkm/benchmarks
* Code for reproducing experiments from our paper can be found in experiments/

## Install
Run ``pip install -e .``

## Demo
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
We have used code and been inspired by https://github.com/PruneTruong/DenseMatching, https://github.com/zju3dv/LoFTR, and https://github.com/GrumpyZhou/patch2pix  

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
