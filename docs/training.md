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
## Megadepth + Scannet
TODO