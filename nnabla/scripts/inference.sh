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

python inference.py \
datasetddpm.h5=./checkpoint/diffusion_models/ADM_FFHQ256_ema_param.h5 \
datasetddpm.config=./guide_config/yaml/config_gmulti.yaml \
datasetddpm.steps=[50,150,250] \
datasetddpm.ema=True \
datasetddpm.dim=[256,256,8448] \
datasetddpm.model_num=10 \
datasetddpm.share_noise=True \
datasetddpm.use_bn=True \
datasetddpm.testing_path=../examples/nnabla/img00000829_img.png \
datasetddpm.output_dir=./checkpoint/pixel_classifier/ffhq_gmulti

