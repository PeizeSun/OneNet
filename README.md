## OneNet: Towards End-to-End One-Stage Object Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](onenet.jpeg)

Comparisons of different label assignment methods. H and W are height and width of feature map, respectively,
K is number of object categories. Previous works on one-stage object detection assign labels by only position cost, such
as (a) box IoU or (b) point distance between sample and ground-truth. In our method, however, (c) classification cost is
additionally introduced. We discover that **classification cost is the key to the success of end-to-end**. Without classification
cost, only location cost leads to redundant boxes of high confidence scores in inference, making NMS post-processing a
necessary component.

## Introduction
[OneNet: Towards End-to-End One-Stage Object Detection](http://arxiv.org/abs/2012.05780)

## Updates
- (11/12/2020) Higher Performance for OneNet is reported by disable gradient clip.

## Comming
  - [ ] Provide models and logs
  - [ ] Support to caffe, onnx, tensorRT
  - [ ] Support to MobileNet
  
## Models
We provide two models 
- dcn is for high accuracy
- nodcn is for easy deployment.

Method | inf_time | train_time | box AP | download
--- |:---:|:---:|:---:|:---:
[R18_dcn](projects/OneNet/configs/onenet.res18.dcn.yaml)     | 109 FPS | 20h  | 29.5 | [model](https://drive.google.com/drive/folders/1LnHMj7pkJhODeZTNHW-UcUZxybKbQmTB)
[R18_nodcn](projects/OneNet/configs/onenet.res18.nodcn.yaml) | 138 FPS | 13h  | 27.5 | [model](https://drive.google.com/drive/folders/1LnHMj7pkJhODeZTNHW-UcUZxybKbQmTB)
[R50_dcn](projects/OneNet/configs/onenet.res50.dcn.yaml)     | 67 FPS  | 36h  | 35.7 | 
[R50_nodcn](projects/OneNet/configs/onenet.res50.nodcn.yaml) | 73 FPS  | 29h  | 32.7 | 

Models are available in [Baidu Drive](https://pan.baidu.com/s/1f0lQ63UEBD-qbHTrsD97hA) by code nhr8.

#### Notes
- We observe about 0.3 AP noise.
- The training time and inference time are on 8 NVIDIA V100 GPUs.
- We use the models pre-trained on imagenet using torchvision. And we provide [torchvision's ResNet-18.pkl](https://drive.google.com/drive/folders/1LnHMj7pkJhODeZTNHW-UcUZxybKbQmTB?usp=sharing) model. More details can be found in [the conversion script](tools/convert-torchvision-to-d2.py).

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
python projects/OneNet/train_net.py --num-gpus 8 \
    --config-file projects/OneNet/configs/onenet.res50.dcn.yaml
```

4. Evaluate OneNet
```
python projects/OneNet/train_net.py --num-gpus 8 \
    --config-file projects/OneNet/configs/onenet.res50.dcn.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

5. Visualize OneNet
```    
python demo/demo.py\
    --config-file projects/OneNet/configs/onenet.res50.dcn.yaml \
    --input path/to/images --output path/to/save_images --confidence-threshold 0.4 \
    --opts MODEL.WEIGHTS path/to/model.pth
```

## License

OneNet is released under MIT License.


## Citing

If you use OneNet in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@article{peize2020onenet,
  title   =  {{OneNet}: Towards End-to-End One-Stage Object Detection},
  author  =  {Peize Sun and Yi Jiang and Enze Xie and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv: 2012.05780},
  year    =  {2020}
}

```
