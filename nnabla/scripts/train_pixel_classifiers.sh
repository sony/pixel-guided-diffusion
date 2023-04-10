#!/bin/bash

start=0
end=750
echo "Train Pixel Classifiers..."


# Train g_t
for step in `seq $start $end`
do
    echo $step
    python train_pixel_classifier.py \
    datasetddpm.h5=./checkpoint/diffusion_models/ADM_FFHQ256_ema_param.h5 \ \
    datasetddpm.config=./guide_config/yaml/config_gt.yaml \
    datasetddpm.steps=[$step] \
    datasetddpm.ema=True \
    datasetddpm.dim=[256,256,2816] \
    datasetddpm.model_num=1 \
    datasetddpm.share_noise=False \
    datasetddpm.use_bn=True \
    datasetddpm.training_path=../datasets/ffhq_34/real/train \
    datasetddpm.testing_path=../datasets/ffhq_34/real/test
done


# Train g_multi
python train_pixel_classifier.py \
datasetddpm.h5=./checkpoint/diffusion_models/ADM_FFHQ256_ema_param.h5 \ \
datasetddpm.config=./guide_config/yaml/config_gmulti.yaml \
datasetddpm.steps=[50,150,250] \
datasetddpm.dim=[256,256,6336] \
datasetddpm.model_num=10
