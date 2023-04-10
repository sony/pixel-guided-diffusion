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

mkdir ./checkpoint/
mkdir ./checkpoint/diffusion_models
wget -nc https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/ADM_FFHQ256_ema_param.h5 -O ./checkpoint/diffusion_models/ADM_FFHQ256_ema_param.h5

mkdir ./checkpoint/pixel_classifier
wget -nc https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/datasetddpm/nnabla/ffhq_gmulti.zip -O ./checkpoint/ffhq_gmulti.zip
wget -nc https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/datasetddpm/nnabla/ffhq_gt_noshare.zip -O ./checkpoint/ffhq_gt_noshare.zip

unzip -n ./checkpoint/ffhq_gmulti.zip -d ./checkpoint/pixel_classifier
unzip -n ./checkpoint/ffhq_gt_noshare.zip -d ./checkpoint/pixel_classifier