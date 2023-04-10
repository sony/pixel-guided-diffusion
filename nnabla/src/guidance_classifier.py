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

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla_diffusion.config.python.datasetddpm import DatasetDDPMConfig
from nnabla_diffusion.ddpm_segmentation.pixel_classifier import pixel_classifier


class GuidanceClassifier(pixel_classifier):
    def __init__(self, conf: DatasetDDPMConfig, timestep_map):
        super(GuidanceClassifier, self).__init__(conf)
        self.model_num = conf.model_num
        self.timestep_map = timestep_map

    def predict_mean_logits(self, features, t):
        batch_size = 1
        if len(features.shape) == 3:
            batch_size = features.shape[0]
            features = features.reshape([-1, self.dim[-1]])

        seg_mode_ensemble = []
        with nn.auto_forward():
            for i in range(self.model_num):
                preds = self.classifier(features, t, i, test=True)
                preds_batch = preds.reshape([1, batch_size, -1, preds.shape[-1]])
                seg_mode_ensemble.append(preds_batch)

            if self.model_num > 1:
                seg_mode_mean = F.mean(F.concatenate(*seg_mode_ensemble, axis=0), axis=0)
                return seg_mode_mean
            else:
                return preds
