# DKM - Deep Kernelized Dense Geometric Matching
Contains code for [*Deep Kernelized Dense Geometric Matching*](https://arxiv.org/abs/2202.00667) 

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

## Downloading Benchmarks
Benchmarking datasets for geometry estimation can be somewhat cumbersome to download. We provide instructions for the benchmarks we use below, and are happy to answer any questions.
### HPatches
First, make sure that the "data/hpatches" path exists, e.g. by

`` ln -s place/where/your/datasets/are/stored/hpatches data/hpatches `` 

Then run (if you don't already have hpatches downloaded)

`` bash scripts/download_hpatches.sh``
### Yfcc100m (OANet Split)
1. We use the split introduced by OANet, the split is described in further detail in https://github.com/PruneTruong/DenseMatching. The resulting directory structure should be data/yfcc100m_test with subdirectories pairs and yfcc100m.
2. Furthermore we use the pair info provided by SuperGlue, https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/yfcc_test_pairs_with_gt.txt . This one should be put in data/yfcc100m_test/yfcc_test_pairs_with_gt.txt

### Megadepth-1500 (LoFTR Split)
1. We use the split made by LoFTR, which can be downloaded here https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc. (You can also use the preprocessed megadepth dataset if you have it available)
2. The images should be located in data/megadepth/Undistorted_SfM/0015 and 0022.
3. The pair infos are provided here https://github.com/zju3dv/LoFTR/tree/master/assets/megadepth_test_1500_scene_info 
3. Put those files in data/megadepth/xxx

### Scannet-1500 (SuperGlue Split)
We use the same split of scannet as superglue.
1. LoFTR provides the split here: https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc
2. Note that ScanNet requires you to sign a License agreement, which can be found http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf
3. This benchmark should be put in the data/scannet folder
## Training
Here we provide instructions for how to train our models, including download of datasets.
### MegaDepth
First the MegaDepth dataset needs to be downloaded and preprocessed. This can be done by the following steps:
1. Download MegaDepth from here: https://www.cs.cornell.edu/projects/megadepth/
2. Extract and preprocess: See https://github.com/mihaidusmanu/d2-net
3. Download our prepared scene info from here: https://github.com/Parskatt/storage/releases/download/prep_scene_info/prep_scene_info.tar
4. File structure should be data/megadepth/phoenix, data/megadepth/Undistorted_SfM, data/megadepth/prep_scene_info.
Then run 
``` bash
python experiments/train_mega.py --gpus 4
```

### MegaDepth + Synthetic
First do the above steps.
For the synthetic dataset, we used CityScapes, ADE20k, and COCO2014.

1. Download Cityscapes(leftImg8bit), ADE20k, and COCO2014 and untar them directly in data/homog_data. This should give you a COCO2014, leftImg8bit, and ADE20K_2016_07_26 folder. In the same place should also be the ade20k_filenames.json and ade20k_folders.json.

``` bash
python experiments/train_mega_synthetic.py --gpus 4
```

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
