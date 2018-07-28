#!/bin/sh
datasetdir=H2FOOD_MINI/train
#datasetdir=VOC2007/trainval
. env/bin/activate
flow --model cfg/yolo-h2food.cfg \
    --train \
    --summary summary \
    --load bin/yolo.weights \
    --annotation $datasetdir/Annotations/ \
    --dataset $datasetdir/JPEGImages/ \
    --labels labels-h2food.txt \
    --threshold 0.1 \
    --epoch 40 \
    --batch 12 \
    --gpu 1.0
