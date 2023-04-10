# Copyright 2022 Sony Group Corporation.
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

from dataclasses import dataclass
from typing import Union, List

from omegaconf import MISSING


@dataclass
class GuidanceConfig:

    # data / weight
    data_dir: str = "./examples"
    classifier_path: str = "./checkpoint/pixel_classifier/ffhq_gt_respacing_1"
    roi_cls_file: str = "./examples/label_edit.json"

    # config for evaluation
    classifier_path_eval: str = "./checkpoint/pixel_classifier/ffhq_gmulti"

    # implementation
    exp_dir: str = "./examples"
    max_steps: int = 750
    batch_size: int = 4
    use_ddim: bool = False
    seed: int = 0
    dilate: int = 3
    seg_scale: float = 100

    image_size: int = 256
    output_dir: str = "./outputs"


# expose config to enable loading from yaml file
from nnabla_diffusion.config.python.utils import register_config
register_config(name="seg_guide", node=GuidanceConfig)
