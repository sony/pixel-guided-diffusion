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

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False  --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --diffusion_steps 1000"
SAMPLE_FLAGS="--use_ddim False"

## FFHQ
DATASET=ffhq_34 # Available datasets: bedroom_28, ffhq_34, cat_15, horse_21
echo "Pixel-wise Guidance..."

RESPACE_STEP=1000

## Quality preserved sampling
# If you perform accelerated sampling, you should use reduced steps for RESPACE_STEP (e.g. RESPACE_STEP=100)
# For accelearated sampling, it is better to set a smaller value of --seg_scale. 

python pixel_wise_guidance.py \
--timestep_respacing $RESPACE_STEP \
--batch_size 4 \
--max_steps 750 \
--seg_scale 1.0 \
--exp_dir ./outs \
--exp experiments/${DATASET}/ddpm_guidance.json \
--edit_map_path ../examples/torch \
--evaluate_weight_path ./checkpoint/pixel_classifiers/ffhq_gmulti/50_150_250_5_6_7_8_12 \
--roi_cls_file ../examples/torch/label_edit.json \
$MODEL_FLAGS \
$SAMPLE_FLAGS


