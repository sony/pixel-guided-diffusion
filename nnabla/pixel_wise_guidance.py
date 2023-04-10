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

import gc
import json
import os
import random
import time

import hydra
import nnabla as nn
import nnabla.functions as F
import nnabla_diffusion.config as config
import numpy as np
from neu.misc import init_nnabla
from nnabla.logger import logger
from nnabla.utils.image_utils import imread, imsave
from nnabla_diffusion.ddpm_segmentation.data_util import get_palette
from nnabla_diffusion.ddpm_segmentation.model import FeatureExtractorDDPM
from nnabla_diffusion.ddpm_segmentation.pixel_classifier import pixel_classifier, save_predictions
from PIL import Image

import guide_config
from src.datasets import SegGuidanceDataIterator
from src.feature_extractor import SegmentationGuidanceDDPM
from src.utils import extract_roi, fix_seed, load_guidance_models, roi_save


def evaluate(conf_dict, output, model_kwargs, editname):
    logger.info("== evaluation ==")
    palette = get_palette(conf_dict["conf"].datasetddpm.category)
    eval_datasetddpm_dim = [256, 256, 8448]
    raw_respacing = conf_dict["loaded_conf"].diffusion.respacing_step
    conf_dict["loaded_conf"].diffusion.respacing_step = 1
    suffix = "_".join([str(step) for step in conf_dict["conf"].datasetddpm.steps])

    model = FeatureExtractorDDPM(
        datasetddpm_conf=conf_dict["conf"].datasetddpm,
        diffusion_conf=conf_dict["loaded_conf"].diffusion,
        model_conf=conf_dict["loaded_conf"].model,
    )

    classifier = pixel_classifier(conf=conf_dict["conf"].datasetddpm)
    classifier.load_ensemble(conf_dict["conf"].seg_guide.classifier_path_eval)
    classifier.model_num = 1

    if conf_dict["conf"].datasetddpm.share_noise:
        np.random.seed(seed=conf_dict["conf"].datasetddpm.seed)
        noise = np.random.rand(1, conf_dict["conf"].datasetddpm.image_size, conf_dict["conf"].datasetddpm.image_size, 3)
    else:
        noise = None

    y = model_kwargs["y"].d.reshape([-1, 1])
    roi_mask = model_kwargs["roi"].d.reshape(-1)
    # predict for each samples in mini-batch
    img_paths, imgs, preds, gts = [], [], [], []

    for ind, sample in enumerate(output):
        with nn.auto_forward():
            if len(sample.shape) != 4:
                sample = sample.reshape([1, *sample.shape])
            imgs.append((sample + 1) * 127.5)

            activations = model.extract_features(sample, noise=noise)
            features = model.collect_features(activations)

            (feature_size, d) = features.d.shape[:2]
            features = F.transpose(F.transpose(features, (1, 0, 2, 3)).reshape([d, -1]), (1, 0))
            # features.forward()
            pred = classifier.predict_labels(conf_dict["conf"], suffix, features, test=True).reshape(-1)
            preds.append(pred.reshape(256, 256))
            gts.append(y)
            pred_roi = pred.reshape(-1)[np.where(roi_mask == 1)[0]].astype(np.uint8)
            y_roi = y.reshape(-1)[np.where(roi_mask == 1)[0]].astype(np.uint8)

            roi_num = len(np.where(roi_mask == 1)[0])

            accuracy = len(np.where(pred_roi == y_roi)[0]) / roi_num

            logger.info("roi_num:{}, accuracy:{}".format(roi_num, accuracy))
    save_predictions(conf_dict["conf"], preds, gts, imgs, editname)

    conf_dict["loaded_conf"].diffusion.respacing_step = raw_respacing


def edit_by_guidance(conf_dict, data_iterator, edit_cls_dict):
    runtime_dict = dict()
    dataset_num = data_iterator._size
    logger.info(f"== Dataset Length: {dataset_num} ==")
    B = conf_dict["conf"].seg_guide.batch_size
    (H, W, C) = conf_dict["loaded_conf"].model.image_shape

    logger.info("== Creating Guidance Model ==")
    feature_extractor = SegmentationGuidanceDDPM(
        guidance_conf=conf_dict["conf"].seg_guide,
        datasetddpm_conf=conf_dict["conf"].datasetddpm,
        diffusion_conf=conf_dict["loaded_conf"].diffusion,
        model_conf=conf_dict["loaded_conf"].model,
    )

    for _ in range(dataset_num):
        gc.collect()
        img, label, label_edit, editname = data_iterator.next()
        if isinstance(editname, np.ndarray):
            editname = str(editname[0])
        cls_ids = edit_cls_dict[editname]

        dilate_roi, pixel_num = extract_roi(conf_dict["conf"], cls_ids, label, label_edit)
        roi_save(conf_dict["conf"], label, label_edit, dilate_roi, editname)

        logger.info("== editname: " + editname + " ==")

        feature_extractor.org_img = np.tile(img, (B, 1, 1, 1))

        start_time = time.time()
        logger.info("== Generating Noise by Deterministic ==")


        X_T = feature_extractor.image_to_noise(
            shape=[
                1,
            ]
            + conf_dict["loaded_conf"].model.image_shape,
            x_init=img,
            use_ema=conf_dict["conf"].datasetddpm.ema,
            progress=True,
        )


        if B > 1:
            X_T = np.tile(X_T, (B, 1, 1, 1))
   

        model_kwargs = dict()
        model_kwargs["y"] = nn.Variable.from_numpy_array(np.squeeze(label_edit))
        model_kwargs["roi"] = nn.Variable.from_numpy_array(dilate_roi)
        model_kwargs["pixels"] = len(np.where(dilate_roi == 1)[0])
        feature_extractor.model_kwargs = model_kwargs

        logger.info("== Edit with pixel wise guidance ==")
        output = feature_extractor.guidance_sample_fn(
            shape=[
                B,
            ]
            + conf_dict["loaded_conf"].model.image_shape,
            x_init=X_T,
            auto_forward=True,
            model_kwargs=model_kwargs,
            use_ema=conf_dict["conf"].datasetddpm.ema,
            progress=True,
        )

        editing_time = time.time() - start_time
        logger.info("== editing time: {} ==".format(editing_time))
        runtime_dict[editname] = editing_time
        evaluate(conf_dict, output[0].d, model_kwargs, editname)

        del img, X_T, dilate_roi, label, label_edit, output


@hydra.main(version_base=None, config_path="guide_config/yaml/", config_name="config_seg_guide")
def main(conf: guide_config.SegmentationGuidanceConfig):

    comm = init_nnabla(ext_name="cudnn", device_id=conf.runtime.device_id, type_config="float", random_pseed=True)

    loaded_conf: config.LoadedConfig = config.load_saved_conf(conf.datasetddpm.config)

    conf_dict = {"conf": conf, "loaded_conf": loaded_conf}

    data_config = config.DatasetConfig(
        name=conf.datasetddpm.category,
        batch_size=1,
        dataset_root_dir=conf.seg_guide.data_dir,
        image_size=loaded_conf.model.image_shape[:-1],
    )
    data_iterator = SegGuidanceDataIterator(conf=data_config, comm=comm)

    nn.parameter.load_parameters(conf.datasetddpm.h5)
    runtime_dict = dict()
    diffusion_process = "ddim" if conf.seg_guide.use_ddim else "ddpm"
    logger.info(f"== Diffusion Process: {diffusion_process} ==")

    load_guidance_models(conf_dict)

    with open(conf.seg_guide.roi_cls_file, "r") as f:
        edit_cls_dict = json.load(f)

    edit_by_guidance(conf_dict, data_iterator, edit_cls_dict)


if __name__ == "__main__":
    fix_seed()
    main()
