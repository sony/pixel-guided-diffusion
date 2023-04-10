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

from tqdm import tqdm
import json
import os
import hydra
import sys
import numpy as np

import nnabla.functions as F
import nnabla as nn
import nnabla.solvers as S
from nnabla.logger import logger

import nnabla_diffusion.config as config
from nnabla_diffusion.ddpm_segmentation.model import FeatureExtractorDDPM
from nnabla_diffusion.ddpm_segmentation.pixel_classifier import pixel_classifier, save_predictions
from nnabla_diffusion.ddpm_segmentation.data_util import get_class_names, get_palette
from nnabla_diffusion.ddpm_segmentation.utils import colorize_mask, to_labels

from src.datasets import InferenceDataIterator
from neu.misc import init_nnabla
from nnabla.utils.image_utils import imread, imsave


def evaluation(model, conf, comm):

    suffix = "_".join([str(step) for step in conf.datasetddpm.steps])
    data_config = config.DatasetConfig(
        name=conf.datasetddpm.category,
        batch_size=1,
        dataset_root_dir=conf.datasetddpm.testing_path,
        image_size=conf.datasetddpm.dim[:-1],
        shuffle_dataset=False
    )

    test_data_iterator = InferenceDataIterator(
        conf=data_config,
        comm=comm,
    )

    dataset_num = test_data_iterator._size
    if conf.datasetddpm.share_noise:
        np.random.seed(seed=conf.datasetddpm.seed)
        noise = np.random.rand(
            1,
            conf.datasetddpm.image_size,
            conf.datasetddpm.image_size,
            3
        )
    else:
        noise = None

    classifier = pixel_classifier(conf=conf.datasetddpm)
    classifier.load_ensemble(conf.datasetddpm.output_dir)

    img_paths, imgs, preds, gts = [], [], [], []
    for row in tqdm(range(dataset_num)):
        with nn.auto_forward():
            image, _ = test_data_iterator.next()
            if len(image.shape) != 4:
                image = image.reshape([1, *image.shape])
            if isinstance(image, tuple):
                image = image[0]
            path = test_data_iterator._data_source.img_paths[row]
            img_paths.append(path)
            activations = model.extract_features(image, noise=noise)
            features = model.collect_features(activations)

            (feature_size, d) = features.d.shape[: 2]
            features = F.transpose(F.transpose(
                features, (1, 0, 2, 3)).reshape([d, -1]), (1, 0))
            # features.forward()
            pred = classifier.predict_labels(
                conf, suffix, features, test=True).reshape(-1)

            palette = get_palette(conf.datasetddpm.category)

            mask = colorize_mask(pred, palette)  # pred:256,256
            label_convert = to_labels(
                np.expand_dims(mask, 0), palette
            )  # label_convert:256,25

            basename = os.path.splitext(os.path.basename(path))[
                0].split("_")[0]
            if os.path.isdir(conf.datasetddpm.testing_path):
                save_path = conf.datasetddpm.testing_path
            else:
                save_path = os.path.dirname(conf.datasetddpm.testing_path)
            imsave(
                os.path.join(save_path,
                             basename + "_raw.png"), mask.reshape(256, 256, 3)[:, :, ::-1]
            )


@ hydra.main(version_base=None, config_path="./guide_config/yaml/", config_name="config_seg_train")
def main(conf: config.TrainDatasetDDPMScriptsConfig):
    comm = init_nnabla(ext_name="cudnn",
                       device_id=conf.runtime.device_id,
                       type_config="float",
                       random_pseed=True)

    loaded_conf: config.LoadedConfig = config.load_saved_conf(
        conf.datasetddpm.config)

    # model definition
    model = FeatureExtractorDDPM(datasetddpm_conf=conf.datasetddpm,
                                 diffusion_conf=loaded_conf.diffusion,
                                 model_conf=loaded_conf.model)

    assert os.path.exists(
        conf.datasetddpm.h5), f"{conf.datasetddpm.h5} is not found. Please make sure the h5 file exists."
    nn.parameter.load_parameters(conf.datasetddpm.h5)

    output_dir = conf.datasetddpm.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    suffix = "_".join([str(step) for step in conf.datasetddpm.steps])
    pretrained = [
        os.path.exists(os.path.join(output_dir, f"t_{suffix}model_{i}.h5"))
        for i in range(conf.datasetddpm.model_num)
    ]

    evaluation(model, conf, comm)


if __name__ == "__main__":
    main()
