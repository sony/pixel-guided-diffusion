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

# Available datasets: bedroom_28, ffhq_34, cat_15, horse_21, celeba_19, ade_bedroom_30
## FFHQ
DATASET=ffhq_34 
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


start=0
end=750
echo "Train Pixel Classifiers..."


## Train g_t
for step in `seq $start $end`
do
    python train_pixel_classifier.py --steps $step --exp experiments/${DATASET}/ddpm_gt.json  $MODEL_FLAGS
done

# Train g_multi
python train_pixel_classifier.py --exp experiments/${DATASET}/ddpm_multi.json  $MODEL_FLAGS

