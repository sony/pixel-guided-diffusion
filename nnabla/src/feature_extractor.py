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
from nnabla_diffusion.config import DatasetDDPMConfig, DiffusionConfig, ModelConfig
from nnabla_diffusion.ddpm_segmentation.model import FeatureExtractorDDPM

from guide_config import GuidanceConfig
from src.guidance_diffusion import SegmentationGaussianDiffusion


class SegmentationGuidanceDDPM(FeatureExtractorDDPM):
    def __init__(
        self,
        guidance_conf: GuidanceConfig,
        datasetddpm_conf: DatasetDDPMConfig,
        diffusion_conf: DiffusionConfig,
        model_conf: ModelConfig,
    ):
        super(SegmentationGuidanceDDPM, self).__init__(datasetddpm_conf, diffusion_conf, model_conf)

        self.org_img = None
        self.batch_size = guidance_conf.batch_size
        self.diffusion = SegmentationGaussianDiffusion(diffusion_conf, guidance_conf, datasetddpm_conf)
        self.diffusion.model = self._define_model()

    def postprocess_fn(self, sample, t, roi, noise=None):
        if isinstance(self.org_img, np.ndarray):
            self.org_img = nn.Variable.from_numpy_array(self.org_img)
        if isinstance(roi, nn.Variable):
            roi = roi.d
        background_stage_t = self.diffusion.q_sample(self.org_img, t, noise=noise)
        if roi.shape != background_stage_t.d.shape:
            roi_np = roi[np.newaxis, :, :, np.newaxis]
            roi_np = np.tile(roi_np, [self.batch_size, 1, 1, 3])
        else:
            roi_np = roi

        roi = nn.Variable.from_numpy_array(roi_np)
        sample = F.add2(F.mul2(sample, roi), F.mul2(background_stage_t, (1 - roi)))


        return sample

    def image_to_noise(self, shape, x_init=None, model_kwargs=None, use_ema=True, dump_interval=-1, progress=False):

        X_T, _, _ = self.sample(
            shape=shape,
            x_init=x_init,
            model_kwargs=model_kwargs,
            use_ema=use_ema,
            dump_interval=dump_interval,
            progress=progress,
            sampler="ddim_rev",
        )

        return X_T.d

    def guidance_sample_fn(
        self,
        shape,
        *,
        x_init=None,
        auto_forward=True,
        model_kwargs=None,
        use_ema=True,
        dump_interval=-1,
        progress=False,
        sampler="ddpm",
        classifier_free_guidance_weight=None,
        no_grad=True
    ):

        # FIXME : update for ddim version
        samplers = {"ddpm": self.diffusion.p_sample_loop_guidance, "ddim": None}
        loop_func = samplers[sampler]
        with nn.no_grad(no_grad):
            return loop_func(
                model=self._define_model(),
                channel_last=self.model_conf.channel_last,
                shape=shape,
                x_init=x_init,
                postprocess_fn=self.postprocess_fn,
                model_kwargs=model_kwargs,
                dump_interval=dump_interval,
                progress=progress,
                classifier_free_guidance_weight=classifier_free_guidance_weight,
            )
