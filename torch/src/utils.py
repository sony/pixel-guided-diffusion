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

"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from PIL import Image
import numpy as np
import random
import cv2
import torch as th
import os


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def torch_fix_seed(seed=42):
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.use_deterministic_algorithms = True


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    return torch.device("cpu")


def extract_roi(opts, cls_ids, label, label_edit):
    pixel_num = {"before_edit": 0, "after_edit": 0, "whole": 0}
    roi = np.zeros((256, 256), np.uint8)
    label = label.view(256, 256).numpy()
    label_edit = label_edit.view(256, 256).numpy()

    for ids in cls_ids:
        roi += (label == int(ids)).astype(np.uint8)
        pixel_num["before_edit"] += int(sum((label == int(ids)).reshape(-1)))
    for ids in cls_ids:
        roi += (label_edit == int(ids)).astype(np.uint8)
        pixel_num["after_edit"] += int(sum((label_edit ==
                                       int(ids)).reshape(-1)))
    roi = roi > 0
    pixel_num["whole"] = int(sum(roi.reshape(-1) > 0))
    if opts["dilate"]:
        kernel = np.ones((opts["dilate"], opts["dilate"]), np.uint8)
        dilate_roi = cv2.dilate(np.float32(roi), kernel,
                                iterations=3).astype(np.uint8)

        return dilate_roi, pixel_num
    else:
        roi = np.float32(roi).astype(np.uint8)
        return roi, pixel_num
