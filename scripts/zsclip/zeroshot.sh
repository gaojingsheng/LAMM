#!/bin/bash

#cd ../..

# custom config
DATA="/data/dataset/ImageClassification"
TRAINER=ZeroshotCLIP
# DATASET=$1
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16
for DATASET in imagenet sun397 caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101
do
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}/${DATASET} \
    --eval-only
done        