# Sparse Spatial Transformers for Few-Shot Learning
This code implements the Sparse Spatial Transformers for Few-Shot Learning(SSFormers).
## Prerequisites
* Linux
* Python 3.7
* Pytorch 1.0+
* GPU + CUDA CuDNN
* pillow, torchvision, scipy, numpy
## Datasets
**Dataset download link:**
* [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view) It contains 100 classes with 600 images in each class, which are built upon the ImageNet dataset. The 100 classes are divided into 64, 16, 20 for meta-training, meta-validation and meta-testing, respectively.
* [tieredImageNet](https://drive.google.com/drive/folders/163HGKZTvfcxsY96uIF6ILK_6ZmlULf_j?usp=sharing)
TieredImageNet is also a subset of ImageNet, which includes 608 classes from 34 super-classes. Compared with miniImageNet, the splits of meta-training(20), meta-validation(6) and meta-testing(8) are set according to the super-classes to enlarge the domain difference between training and testing phase. The dataset also include more images for training and evaluation (779,165 images in total).

**Note: You need to manually change the dataset directory.**

## Few-shot Classification
* Train a 5-way 1-shot SSFormers model based on Conv-64F (on miniImageNet dataset):
```
 python experiments/run_trainer.py  --cfg ./configs/miniImagenet/ST_N5K1_R12.yaml --device 0
```
Test model on the test set:
```
python experiments/run_evaluator.py --cfg ./configs/miniImagenet/ST_N5K1_R12.yaml -c ./checkpoint/*/*.pth --device 0
```
and semi-supervised few-shot learning tasks (with trial t=1).
```
python experiments/run_semi_trainer.py --cfg ./configs/miniImagenet/ST_N5K1_semi_with_extractor.yaml --device 0 -t 1

python experiments/run_semi_evaluator.py --cfg ./configs/miniImagenet/ST_N5K1_semi_with_extractor.yaml -c ./checkpoints/*/*.pth --device 0
```
