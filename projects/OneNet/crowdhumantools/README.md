## Evaluation on CrowdHuman
The evalution code is built on top of [cvpods](https://github.com/Megvii-BaseDetection/cvpods).


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

#### Notes
- The default evaluation code in training should be ignored, since it only considers at most 100 objects in one image, while crowdhuman image contains more than 100 objects.