#!/bin/bash

DATA="/path/to/dataset/folder"
TRAINER=CoOp

CFG=vit_b16_ep50_ctxv1

for SHOTS in 16 
do
    for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 sun397 imagenet 
    do
        for SEED in  1 2 3
        do
            DIR=output/base2new_set2/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
            if [ -d "$DIR" ]; then
                echo "Results are available in ${DIR}. Resuming..."
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --shots ${SHOTS}\
                --triplet-loss \
                --origin-clip \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES new
            else
                echo "Run this job and save the output to ${DIR}"
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --shots ${SHOTS}\
                --triplet-loss \
                --origin-clip \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES new
            fi
        done
    done      
done  