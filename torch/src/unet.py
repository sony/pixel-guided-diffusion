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

import torch as th
from ddpm_segmentation.guided_diffusion.guided_diffusion.unet import UNetModel
from ddpm_segmentation.guided_diffusion.guided_diffusion.nn import timestep_embedding


class UNetMultipleOutputModel(UNetModel):
    def __init__(self, extract_blocks, **kwargs):
        super().__init__(**kwargs)
        self.extract_blocks = extract_blocks

    def forward(self, x, timesteps, y=None, roi=None, ref_roi=None, ref_x=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        hs = []
        extract_output = []
        emb = self.time_embed(timestep_embedding(
            timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for idx, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)

            h = module(h, emb)
            if idx in self.extract_blocks:
                extract_output.append(h)

        h = h.type(x.dtype)
        out = self.out(h)
        return out, extract_output
