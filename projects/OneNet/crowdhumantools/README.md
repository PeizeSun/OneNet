## Evaluation on CrowdHuman

Method | inf_time | train_time | AP50 | mMR | recall | download
--- |:---:|:---:|:---:|:---:|:---:|:---:
[R50_RetinaNet](projects/OneNet/configs/onenet.retinanet.res50.crowdhuman.yaml) | 26 FPS  | 11.5h | 90.9 | 48.8 | 98.0 |[model](https://drive.google.com/drive/folders/1LnHMj7pkJhODeZTNHW-UcUZxybKbQmTB) \| [log](https://drive.google.com/drive/folders/1LnHMj7pkJhODeZTNHW-UcUZxybKbQmTB)
[R50_FCOS](projects/OneNet/configs/onenet.fcos.res50.crowdhuman.yaml) | 27 FPS  | 4.5h  | 90.6 | 48.6 | 97.7 | [model](https://drive.google.com/drive/folders/1LnHMj7pkJhODeZTNHW-UcUZxybKbQmTB) \| [log](https://drive.google.com/drive/folders/1LnHMj7pkJhODeZTNHW-UcUZxybKbQmTB)

Models are available in [Baidu Drive](https://pan.baidu.com/s/1f0lQ63UEBD-qbHTrsD97hA) by code nhr8.

#### Notes
- The evalution code is built on top of [cvpods](https://github.com/Megvii-BaseDetection/cvpods).
- The default evaluation code in training should be ignored, since it only considers at most 100 objects in one image, while crowdhuman image contains more than 100 objects.
- The training time and inference time are on 8 NVIDIA V100 GPUs. We observe the same type of GPUs in different clusters may cost different time.
- More training steps are in the [crowdhumantools](https://github.com/PeizeSun/OneNet/tree/main/projects/OneNet/crowdhumantools).

#### Installation

1. Convert crowdhuman annotations of odgt format to coco format. More details can be found in [the conversion script](convert_crowdhuman_to_coco.py).

2. Link crowdhuman dataset path to OneNet/datasets/crowdhuman
```
mkdir -p datasets/crowdhuman
ln -s /path_to_crowdhuman_dataset/annotations datasets/coco/annotations
ln -s /path_to_crowdhuman_dataset/CrowdHuman_train datasets/coco/CrowdHuman_train
ln -s /path_to_crowdhuman_dataset/CrowdHuman_val datasets/crowdhuman/CrowdHuman_val
```

3. Train on crowdhuman
```
python projects/OneNet/train_net.py --num-gpus 8 \
    --config-file projects/OneNet/configs/onenet.fcos.res50.crowdhuman.yaml
```

4. Evaluate on crowdhuman
```
python projects/OneNet/train_net.py --num-gpus 8 \
    --config-file projects/OneNet/configs/onenet.fcos.res50.crowdhuman.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
python projects/OneNet/crowdhumantools/crowdhuman_eval.py \
    --result output_onenet_r50fcos_crowdhuman/inference/coco_instances_results.json \
    --gt datasets/crowdhuman/annotations/val.json
```
