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

import sys
import torch
from torch import nn
from typing import List
from src.utils import dev
from ddpm_segmentation.src.feature_extractors import FeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"


def create_feature_extractor(model_type, **kwargs):
    """Create the feature extractor for <model_type> architecture."""
    if model_type == "ddpm":
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    elif model_type == "guidance_ddpm":
        print("Creating Guidance DDPM Feature Extractor...")
        feature_extractor = GuidanceFeatureExtractorDDPM(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor


def save_tensors(module: nn.Module, features, name: str):
    """Process and save activations in the module."""
    if type(features) in [list, tuple]:
        features = [
            f.detach().float() if f is not None else None for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, "activations")
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], "activations")
    return out


class FeatureExtractorDDPM(FeatureExtractor):
    """
    Wrapper to extract features from pretrained DDPMs.

    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    """

    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.input = []

        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect

        from ddpm_segmentation.guided_diffusion.guided_diffusion.script_util import (
            create_model_and_diffusion,
        )

        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(
            **expected_args)

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(dev())
        if kwargs["use_fp16"]:
            self.model.convert_to_fp16()
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):

        activations = []
        input_noisy_x = []
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            input_noisy_x.append(noisy_x.cpu().data.numpy())
            self.model(noisy_x, self.diffusion._scale_timesteps(t))
            # Extract activations
            for block in self.feature_blocks:

                activations.append(block.activations)
                block.activations = None
        self.input.append(input_noisy_x)

        # Per-layer list of activations [N, C, H, W]
        return activations


class GuidanceFeatureExtractorDDPM(FeatureExtractor):
    """
    Wrapper to extract features from pretrained DDPMs.

    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    """

    def __init__(self, blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.input = []

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        from src.script_util_segguide import create_model_and_segdiffusion
        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(
            create_model_and_segdiffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_segdiffusion(
            **expected_args)

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(dev())
        if kwargs["use_fp16"]:
            self.model.convert_to_fp16()
        self.model.eval()

    def forward(self, x, t):
        activations = []
        t = torch.tensor([t]).to(x.device)
        model_output, activations = self.model(
            x, self.diffusion._scale_timesteps(t))

        # Per-layer list of activations [N, C, H, W]
        return activations
