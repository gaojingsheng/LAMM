#!/bin/bash

DATA="/path/to/dataset/folder"
TRAINER=CoOp
CFG=vit_b16_ep50_ctxv1
LOADEP=50

for SHOTS in 16 
do
    for DATASET in imagenet-sketch imagenetv2
    do
        for SEED in 1 2 3
        do
            DIR=output/cross/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
            if [ -d "$DIR" ]; then
                echo "Results are available in ${DIR}. Resuming..."
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --model-dir output/base/imagenet/shots_16/LAMM/vit_b16_ep50_ctxv1/seed${SEED} \
                --output-dir ${DIR} \
                --load-epoch ${LOADEP} \
                --triplet-loss \
                --eval-only \
                DATASET.NUM_SHOTS ${SHOTS} 
            else
                echo "Run this job and save the output to ${DIR}"
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --model-dir output/base/imagenet/shots_16/LAMM/vit_b16_ep50_ctxv1/seed${SEED} \
                --output-dir ${DIR} \
                --load-epoch ${LOADEP} \
                --triplet-loss \
                --eval-only \
                DATASET.NUM_SHOTS ${SHOTS} 
            fi
        done
    done 
done       