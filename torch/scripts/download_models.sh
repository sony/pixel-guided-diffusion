# Copyright 2023 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mkdir ./checkpoint
mkdir ./checkpoint/diffusion_model
wget -nc https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/models/ddpm_checkpoints/ffhq.pt -O ./checkpoint/diffusion_model/ffhq.pt

mkdir ./checkpoint/pixel_classifiers
wget -nc https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/datasetddpm/torch/ffhq_gmulti.zip -O ./checkpoint/ffhq_gmulti.zip
wget -nc https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/datasetddpm/torch/ffhq_gt_750.zip -O ./checkpoint/ffhq_gt_750.zip

unzip -n ./checkpoint/ffhq_gmulti.zip -d ./checkpoint/pixel_classifiers
unzip -n ./checkpoint/ffhq_gt_750.zip -d ./checkpoint/pixel_classifiers