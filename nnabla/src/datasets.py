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
from glob import glob
from typing import List

import numpy as np
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.image_utils import imread, imresize
from nnabla_diffusion.config import DatasetConfig
from nnabla_diffusion.dataset.common import SimpleDatasource, _resize_img, resize_center_crop, resize_random_crop
from nnabla_diffusion.ddpm_segmentation.data_util import get_palette


def SegGuidanceDataIterator(conf: DatasetConfig, comm, rng=None):
    logger.info("== Loading Dataset ==")
    img_path = sorted(glob(os.path.join(conf.dataset_root_dir, "*_img.png")))
    label_path = sorted(glob(os.path.join(conf.dataset_root_dir, "*_raw.png")))
    label_edit_path = sorted(glob(os.path.join(conf.dataset_root_dir, "*_edited.png")))

    for img, label, label_edit in zip(img_path, label_path, label_edit_path):
        checker = set([os.path.basename(imgname).split("_")[0] for imgname in [img, label, label_edit]])
        if len(checker) != 1:
            logger.info("== Invalid Pair Dataset ==")
            break

    ds = SegGuideDataSource(
        conf, img_paths=img_path, labels=label_path, labels_edit=label_edit_path, category=conf.name, rng=rng
    )

    return data_iterator(ds, conf.batch_size, with_memory_cache=False, use_thread=True, with_file_cache=False)


class SegGuideDataSource(SimpleDatasource):
    def __init__(
        self,
        conf: DatasetConfig,
        img_paths: List[str],
        *,
        labels: List[str],
        labels_edit: List[str],
        category="ffhq_34",
        rng=None
    ):
        super(SegGuideDataSource, self).__init__(conf, img_paths=img_paths, labels=labels, rng=rng)

        self._variables = ["image", "label", "label_edit", "editname"]
        self.labels_edit = labels_edit
        self.palette = get_palette(category)
        self.resolution = conf.image_size[0]

    def _to_labels(self, masks, palette):
        results = np.zeros((256, 256), dtype=np.int32).reshape(-1)
        mask_flat = masks.reshape(-1, 3)  # 256*256, 3

        label = 0
        num = 0
        for color in np.array(palette).reshape(-1, 3):
            idxs = np.where((mask_flat == color).all(-1))
            results[idxs] = label
            label += 1
        return results.reshape(256, 256)

    def _get_data(self, i):
        image_idx = self._indexes[i]

        # keep data paths
        if self.data_history.full():
            self.data_history.get()
        self.data_history.put(self.img_paths[image_idx])

        if self.on_memory and self.images[image_idx] is not None:
            return (self.images[image_idx], label)

        img = imread(self.img_paths[image_idx], channel_first=not self.channel_last, num_channels=3)
        label = imread(self.labels[image_idx], channel_first=False, num_channels=3)
        label_edit = imread(self.labels_edit[image_idx], channel_first=False, num_channels=3)

        editname = os.path.splitext(os.path.basename(self.labels_edit[image_idx]))[0]


        mask_tensor = []
        for label in [label, label_edit]:
            label_rgb = label[:, :, ::-1]
            mask = self._to_labels(label_rgb, self.palette).astype("uint8")
            mask = imresize(mask, (self.resolution, self.resolution), interpolate="nearest")

            mask_tensor.append(mask)

        if self.fix_aspect_ratio:
            # perform resize and crop to keep original aspect ratio.

            if self.random_crop:
                # crop randomly so that cropped size equals to self.im_size
                img = resize_random_crop(img, self.im_size[0], channel_first=not self.channel_last)
            else:
                # always crop the center region of an image
                img = resize_center_crop(img, self.im_size[0], channel_first=not self.channel_last)
        else:
            # Breaking original aspect ratio, forcely resize image to self.im_size.
            img = _resize_img(img, self.im_size[0], channel_first=not self.channel_last, fix_aspect_ratio=False)

        # rescale pixel intensity to [-1, 1]
        img = img / 127.5 - 1

        if self.on_memory:
            self.images[image_idx] = img

        return img, mask_tensor[0], mask_tensor[1], editname


def InferenceDataIterator(conf: DatasetConfig, comm, rng=None):
    logger.info("== Loading Dataset ==")
    if not os.path.isdir(conf.dataset_root_dir):
        img_path = [conf.dataset_root_dir]
    else:
        img_path = sorted(glob(os.path.join(conf.dataset_root_dir, "*_img.png")))

    ds = InferenceDataSource(conf, img_paths=img_path, category=conf.name, rng=rng)

    return data_iterator(ds, conf.batch_size, with_memory_cache=False, use_thread=True, with_file_cache=False)


class InferenceDataSource(SimpleDatasource):
    def __init__(self, conf: DatasetConfig, img_paths: List[str], *, category="ffhq_34", rng=None):
        super(InferenceDataSource, self).__init__(conf, img_paths=img_paths, rng=rng)

        self._variables = ["image", "label"]
        self.palette = get_palette(category)
        self.resolution = conf.image_size[0]

    def _to_labels(self, masks, palette):
        results = np.zeros((256, 256), dtype=np.int32).reshape(-1)
        mask_flat = masks.reshape(-1, 3)  # 256*256, 3

        label = 0
        num = 0
        for color in np.array(palette).reshape(-1, 3):
            idxs = np.where((mask_flat == color).all(-1))
            results[idxs] = label
            label += 1
        return results.reshape(256, 256)

    def _get_data(self, i):
        image_idx = self._indexes[i]

        # keep data paths
        if self.data_history.full():
            self.data_history.get()
        self.data_history.put(self.img_paths[image_idx])

        if self.on_memory and self.images[image_idx] is not None:
            return self.images[image_idx]

        img = imread(self.img_paths[image_idx], channel_first=not self.channel_last, num_channels=3)

        if self.fix_aspect_ratio:
            # perform resize and crop to keep original aspect ratio.

            if self.random_crop:
                # crop randomly so that cropped size equals to self.im_size
                img = resize_random_crop(img, self.im_size[0], channel_first=not self.channel_last)
            else:
                # always crop the center region of an image
                img = resize_center_crop(img, self.im_size[0], channel_first=not self.channel_last)
        else:
            # Breaking original aspect ratio, forcely resize image to self.im_size.
            img = _resize_img(img, self.im_size[0], channel_first=not self.channel_last, fix_aspect_ratio=False)

        # rescale pixel intensity to [-1, 1]
        img = img / 127.5 - 1

        if self.on_memory:
            self.images[image_idx] = img

        return img, None
