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

import os
import random

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.logger import logger
from nnabla.utils.image_utils import imread, imsave
from nnabla_diffusion.config.python.datasetddpm import DatasetDDPMConfig
from nnabla_diffusion.ddpm_segmentation.data_util import get_palette
from nnabla_diffusion.ddpm_segmentation.utils import colorize_mask
from tqdm import tqdm


def load_multiple_ensemble(conf):
    logger.info("== Loading pretrained classifiers ==")
    model_num = conf.segmentation.model_num
    suffix = "_".join([str(step) for step in conf.segmentation.steps])
    output_dir = conf.segmentation.output_dir

    for i in tqdm(range(model_num)):
        model_path = os.path.join(output_dir, "t_" + suffix + "model_" + str(i) + ".h5")

        nn.load_parameters(model_path)


def load_guidance_models(conf):
    logger.info("== Loading pretrained classifiers ==")
    model_num = conf["conf"].datasetddpm.model_num
    classifier_path = conf["conf"].seg_guide.classifier_path

    for step in tqdm(range(conf["loaded_conf"].diffusion.t_start)):
        for i in range(model_num):
            model_path = os.path.join(classifier_path, "t_" + str(step) + "model_" + str(i) + ".h5")

            nn.load_parameters(model_path)


def dilate(src, ksize=3):
    h, w = src.shape
    dst = src.copy()
    d = int((ksize - 1) / 2)

    for y in range(0, h):
        for x in range(0, w):
            roi = src[y - d : y + d + 1, x - d : x + d + 1]
            if np.count_nonzero(roi) > 0:
                dst[y][x] = 1
    return dst


def extract_roi(conf, cls_ids, label, label_edit):
    pixel_num = {"before_edit": 0, "after_edit": 0, "whole": 0}
    roi = np.zeros((256, 256), np.uint8)
    label = label.reshape(256, 256)
    label_edit = label_edit.reshape(256, 256)

    for ids in cls_ids:
        roi += (label == int(ids)).astype(np.uint8)
        pixel_num["before_edit"] += int(sum((label == int(ids)).reshape(-1)))
    for ids in cls_ids:
        roi += (label_edit == int(ids)).astype(np.uint8)
        pixel_num["after_edit"] += int(sum((label_edit == int(ids)).reshape(-1)))
    roi = roi > 0
    pixel_num["whole"] = int(sum(roi.reshape(-1) > 0))

    if conf.seg_guide.dilate:
        kernel = np.ones((conf.seg_guide.dilate, conf.seg_guide.dilate), np.uint8)
        for _ in range(3):
            dilate_roi = dilate(np.float32(roi), conf.seg_guide.dilate).astype(np.uint8)
            roi = dilate_roi

        return dilate_roi, pixel_num
    else:
        roi = np.float32(roi).astype(np.uint8)
        return roi, pixel_num


def roi_save(conf, label_org, label_edit, roi_np, editname, alpha=60):
    palette = get_palette(conf.datasetddpm.category)
    shape = label_org.shape
    label_org = np.squeeze(label_org)
    label_edit = np.squeeze(label_edit)
    alpha_ch = np.transpose(np.ones(shape), (1, 2, 0)) * 255
    mask_ = colorize_mask(label_org, palette)
    mask = np.concatenate([colorize_mask(label_org, palette)[:, :, ::-1], alpha_ch], axis=2).reshape(-1, 4)
    edit_mask = np.concatenate([colorize_mask(label_edit, palette)[:, :, ::-1], alpha_ch], axis=2).reshape(-1, 4)

    roi_np = roi_np.reshape(-1)
    mask[np.where(roi_np == 0)[0], -1] = alpha
    edit_mask[np.where(roi_np == 0)[0], -1] = alpha
    if not os.path.exists(conf.datasetddpm.output_dir):
        os.makedirs(conf.datasetddpm.output_dir)

    imsave(
        os.path.join(conf.datasetddpm.output_dir, (editname + "_label_roi.png")),
        mask.reshape(256, 256, 4).astype(np.uint8),
    )
    imsave(
        os.path.join(conf.datasetddpm.output_dir, (editname + "_label_edit_roi.png")),
        edit_mask.reshape(256, 256, 4).astype(np.uint8),
    )


def collect_features(conf: DatasetDDPMConfig, activations):
    resized_activations = []
    # with nn.auto_forward():
    for feats in activations:
        feats = F.transpose(feats, (0, 3, 1, 2))
        feats = F.interpolate(feats, output_size=conf.dim[:-1], mode=conf.upsampling_mode)

        resized_activations.append(feats)

    all_activations = F.concatenate(*resized_activations, axis=1)

    return all_activations


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


