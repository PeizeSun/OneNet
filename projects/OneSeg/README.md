
This is final group project of HKU, COMP9501, accomplished by Peize Sun and Sheng Jin.


## Instance Segmentation

Method | train_time | mask AP | download
--- |:---:|:---:|:---
[R50](configs/oneseg.res50.800size.yaml)    | 20h  | 33.9 | -
[R50_dcn](configs/oneseg.res50.800size.dcn.yaml) | 25h  | 36.5 | - 


## Installation
The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and [DETR](https://github.com/facebookresearch/detr).

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Install and build libs
```
git clone https://github.com/PeizeSun/OneNet.git
cd OneNet
python setup.py build develop
```

2. Link coco dataset path to OneNet/datasets/coco
```
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

3. Train OneNet
```
python projects/OneSeg/train_net.py --num-gpus 8 \
    --config-file projects/OneSeg/configs/oneseg.res50.800size.yaml
```

4. Evaluate OneNet
```
python projects/OneSeg/train_net.py --num-gpus 8 \
    --config-file projects/OneSeg/configs/oneseg.res50.800size.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

