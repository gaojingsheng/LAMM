#!/bin/bash

DATA="/path/to/dataset/folder"
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx

for SHOTS in 16 8 4 2 1
do
    for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 sun397 imagenet
    do
        for SEED in 1 2 3
        do
            DIR=output/base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
            if [ -d "$DIR" ]; then
                echo "Results are available in ${DIR}. Resuming..."
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --triplet-loss \
                DATASET.NUM_SHOTS ${SHOTS} 
            else
                echo "Run this job and save the output to ${DIR}"
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --triplet-loss \
                DATASET.NUM_SHOTS ${SHOTS} 
            fi
        done
    done
done