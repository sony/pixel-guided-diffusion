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

python pixel_wise_guidance.py \
datasetddpm.config=./guide_config/yaml/config_guidance_750.yaml \
datasetddpm.h5=./checkpoint/diffusion_models/ADM_FFHQ256_ema_param.h5 \
datasetddpm.ema=True \
datasetddpm.model_num=1 \
datasetddpm.share_noise=False \
datasetddpm.output_dir=./results \
seg_guide.batch_size=3 \
seg_guide.classifier_path=./checkpoint/pixel_classifier/ffhq_gt_noshare \
seg_guide.seg_scale=40 \
seg_guide.data_dir=../examples/nnabla \
seg_guide.roi_cls_file=../examples/nnabla/label_edit.json
