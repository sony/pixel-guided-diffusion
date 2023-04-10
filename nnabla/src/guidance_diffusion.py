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
from functools import partial

import nnabla as nn
import nnabla.functions as F
import numpy as np
from neu.misc import AttrDict
from nnabla.utils.image_utils import imread, imsave
from nnabla_diffusion.config import DiffusionConfig
from nnabla_diffusion.ddpm_segmentation.data_util import get_palette
from nnabla_diffusion.ddpm_segmentation.utils import colorize_mask
from nnabla_diffusion.diffusion_model.diffusion import GaussianDiffusion, ModelVarType, noise_like
from nnabla_diffusion.diffusion_model.layers import chunk
from nnabla_diffusion.diffusion_model.utils import Shape4D

from src.guidance_classifier import GuidanceClassifier
from src.utils import collect_features


class SegmentationGaussianDiffusion(GaussianDiffusion):
    """
    Extended Diffusion Model for inferring the segmentation map and mean and variance simultaneously
    """

    def __init__(self, conf: DiffusionConfig, guidance_conf, datasetddpm_conf):
        super(SegmentationGaussianDiffusion, self).__init__(conf)
        # FIXME : self
        self.datasetddpm_conf = datasetddpm_conf
        self.classifier = GuidanceClassifier(conf=self.datasetddpm_conf, timestep_map=self.timestep_map)
        self.batch_size = guidance_conf.batch_size
        self.dim = datasetddpm_conf.dim
        self.seg_scale = guidance_conf.seg_scale
        self.num_class = datasetddpm_conf.number_class

        self.ema = datasetddpm_conf.ema
        self.model_kwargs = None

    def build_guidance_graph(self, x, t, y, roi):
        t_rescale = self._rescale_timestep(t)
        int_t = int(t_rescale.d[0]) if len(t_rescale.d) > 1 else int(t_rescale.d)
        with nn.no_grad(False):
            if self.ema:
                with nn.parameter_scope("ema"):
                    pred, activation_list = self.model(x, t_rescale, **{})
            else:
                pred, activation_list = self.model(x, t_rescale, **{})

            features = collect_features(self.datasetddpm_conf, activation_list)
            x_middle = features.reshape([self.batch_size, self.dim[-1], -1])
            x_middle = F.transpose(x_middle, [0, 2, 1]).reshape([-1, self.dim[-1]])

            mean_logits = self.classifier.predict_mean_logits(x_middle, int_t)

            y = y.reshape([-1, 1])
            mean_logits = mean_logits.reshape([self.batch_size, -1, self.num_class])
            loss_ = [F.softmax_cross_entropy(mean_logit, y) for mean_logit in mean_logits]
            softmax_mul = [F.sum(F.mul2(softmax_batch, roi)) for softmax_batch in loss_]
            masked_loss = F.add_n(*softmax_mul)

            return pred, masked_loss, mean_logits

    def resampling_mean(self, preds, gradient_scaled):
        new_mean = preds.mean + preds.var * gradient_scaled
        return new_mean

    def p_mean_var(
        self,
        model,
        x_t,
        t,
        *,
        model_kwargs=None,
        clip_denoised=True,
        channel_last=False,
        classifier_free_guidance_weight=None,
    ):
        """
        Compute mean and var of p(x_{t-1}|x_t) from model.

        Args:
            model (Callable): A callbale that takes x_t and t and predict noise (and more).
            x_t (nn.Variable): The (B, C, ...) tensor at timestep t (x_t).
            t (nn.Variable): A 1-D tensor of timesteps. The first axis represents batchsize.
            clip_denoised (boolean): If True, clip the denoised signal into [-1, 1].
            channel_last (boolean): Whether the channel axis is the last axis of an Array or not.
            classifier_free_guidance_weight (float): A weight for classifier-free guidance.

        Returns:
            An AttrDict containing the following items:
                "mean": the mean predicted by model.
                "var": the variance predicted by model (or pre-defined variance).
                "log_var": the log of "var".
                "xstart": the x_0 predicted from x_t and t by model.
        """
        B, C, H, W = Shape4D(x_t.shape, channel_last=channel_last).get_as_tuple("bchw")
        assert t.shape == (B,)

        if model_kwargs is None:
            model_kwargs = {}

        pred = model(x_t, t, **model_kwargs)
        if isinstance(pred, tuple):
            pred, _ = pred

        if self.model_var_type == ModelVarType.LEARNED_RANGE:
            pred_noise, pred_var_coeff = chunk(pred, num_chunk=2, axis=3 if channel_last else 1)

            min_log = self._extract(self.posterior_log_var_clipped, t, x_t.shape)
            max_log = F.log(self._extract(self.betas, t, x_t.shape))

            # No need to constrain v, according to the "improved DDPM" paper.
            v = pred_var_coeff
            model_log_var = v * max_log + (1 - v) * min_log
            model_var = F.exp(model_log_var)
        else:
            # Model only predicts noise
            pred_noise = pred

            model_log_var, model_var = {
                ModelVarType.FIXED_LARGE: lambda: (
                    self._extract(self.log_betas_clipped, t, x_t.shape),
                    self._extract(self.betas_clipped, t, x_t.shape),
                ),
                ModelVarType.FIXED_SMALL: lambda: (
                    self._extract(self.posterior_log_var_clipped, t, x_t.shape),
                    self._extract(self.posterior_var, t, x_t.shape),
                ),
            }[self.model_var_type]()

        # classifier-free guidance
        if classifier_free_guidance_weight is not None and classifier_free_guidance_weight > 0:
            model_kwargs_uncond = model_kwargs.copy()
            model_kwargs_uncond["cond_drop_rate"] = 1
            pred_uncond = model(x_t, t, **model_kwargs_uncond)

            if self.model_var_type == ModelVarType.LEARNED_RANGE:
                pred_noise_uncond = pred_uncond[..., :3] if channel_last else pred_uncond[:, :3]
            else:
                pred_noise_uncond = pred_uncond

            # (1 + w) * eps(t, c) - w * eps(t)
            w = classifier_free_guidance_weight
            pred_noise = (1 + w) * pred_noise - w * pred_noise_uncond

        x_recon = self.predict_xstart_from_noise(x_t=x_t, t=t, noise=pred_noise)

        if clip_denoised:
            x_recon = F.clip_by_value(x_recon, -1, 1)

        model_mean, _, _ = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)

        assert model_mean.shape == x_recon.shape

        assert model_mean.shape == model_var.shape == model_log_var.shape or (
            model_mean.shape[0] == model_var.shape[0] == model_log_var.shape[0]
            and model_var.shape[1:] == (1, 1, 1)
            and model_log_var.shape[1:] == (1, 1, 1)
        )

        # returns
        ret = AttrDict()
        ret.mean = model_mean
        ret.var = model_var
        ret.log_var = model_log_var
        ret.xstart = x_recon

        return ret

    def p_mean_var_guidance(
        self,
        model,
        x_t,
        t,
        p_bar,
        *,
        model_kwargs=None,
        clip_denoised=True,
        channel_last=False,
        classifier_free_guidance_weight=None,
    ):
        """
        Compute mean and var of p(x_{t-1}|x_t) from model.

        Args:
            model (Callable): A callbale that takes x_t and t and predict noise (and more).
            x_t (nn.Variable): The (B, C, ...) tensor at timestep t (x_t).
            t (nn.Variable): A 1-D tensor of timesteps. The first axis represents batchsize.
            p_bar : progression bar set by tqdm() function.
            clip_denoised (boolean): If True, clip the denoised signal into [-1, 1].
            channel_last (boolean): Whether the channel axis is the last axis of an Array or not.
            classifier_free_guidance_weight (float): A weight for classifier-free guidance.

        Returns:
            An AttrDict containing the following items:
                "mean": the mean predicted by model.
                "var": the variance predicted by model (or pre-defined variance).
                "log_var": the log of "var".
                "xstart": the x_0 predicted from x_t and t by model.
        """
        B, C, H, W = Shape4D(x_t.shape, channel_last=channel_last).get_as_tuple("bchw")
        assert t.shape == (B,)

        if model_kwargs is None:
            model_kwargs = {}

        # calculating gradient for pixel-wise gudance
        with nn.auto_forward():
            x_in = x_t.get_unlinked_variable(need_grad=True)
            x_in.grad.zero()
            pred, masked_loss, mean_logits = self.build_guidance_graph(
                x_in, t, model_kwargs["y"], model_kwargs["roi"].reshape([-1, 1])
            )

            loss = masked_loss.d / model_kwargs["pixels"]
            roi = model_kwargs["roi"].d.reshape(-1)
            y = model_kwargs["y"].d.reshape(-1)[np.where(roi == 1)[0]]
            palette = get_palette(self.datasetddpm_conf.category)
            pred_ = np.argmax(mean_logits.d[0], axis=1)
            mask = colorize_mask(pred_.reshape(256, 256), palette)

            t_rescale = self._rescale_timestep(t)
            int_t = int(t_rescale.d[0]) if len(t_rescale.d) > 1 else int(t_rescale.d)
    
            pred_label = [np.argmax(mean_logit_, axis=1)[np.where(roi == 1)[0]] for mean_logit_ in mean_logits.d]

            accuracy = np.mean(
                [len(np.where(y == pred_l)[0]) / len(np.where(roi == 1)[0]) for pred_l in pred_label]
            )

            masked_loss.forward(clear_no_need_grad=True)
            masked_loss.backward(clear_buffer=True)

            gradient = x_in.grad.data / model_kwargs["pixels"]

            gradient_scaled = nn.Variable.from_numpy_array((-self.seg_scale * gradient))
            postfix = (
                f"masked loss: {loss:.2f}, grad_max: {np.max(gradient_scaled.d):.4f}, accuracy: {accuracy:.5f}"
            )
            p_bar.set_postfix_str(postfix)

        return self.p_mean_var(model=lambda *args, **kwargs: pred, 
                              x_t=x_t, 
                              t=t, 
                              model_kwargs=model_kwargs,
                              clip_denoised=clip_denoised,
                              channel_last=channel_last,
                              classifier_free_guidance_weight=classifier_free_guidance_weight
                            ), gradient_scaled

    def p_sample_guidance(
        self,
        model,
        x_t,
        t,
        *,
        p_bar,
        model_kwargs=None,
        clip_denoised=True,
        noise_function=F.randn,
        repeat_noise=False,
        no_noise=False,
        channel_last=False,
        classifier_free_guidance_weight=None,
        postprocess_fn=None,
    ):
        """
        Sample from the model for one step.
        Also return predicted x_start.
        """

        preds, gradient_scaled = self.p_mean_var_guidance(
            model=model,
            x_t=x_t,
            t=t,
            p_bar=p_bar,
            model_kwargs=model_kwargs,
            clip_denoised=clip_denoised,
            channel_last=channel_last,
            classifier_free_guidance_weight=classifier_free_guidance_weight,
        )

        # no noise when t == 0
        if no_noise:
            return preds.mean, preds.xstart

        preds.mean = self.resampling_mean(preds, gradient_scaled)

        noise = noise_like(x_t.shape, noise_function, repeat_noise)
        assert noise.shape == x_t.shape

        # sample from gaussian N(model_mean, model_var)
        sample = preds.mean + F.exp(0.5 * preds.log_var) * noise

        if postprocess_fn is not None:
            sample = postprocess_fn(sample, t, model_kwargs["roi"])
        return sample, preds.xstart

    def p_sample_loop_guidance(
        self,
        *args,
        channel_last=False,
        postprocess_fn=None,
        classifier_free_guidance_weight=None,
        **kwargs,
    ):
        indices = list(range(self.num_timesteps))[::-1]

        samples = []
        pred_x_starts = []

        from tqdm.auto import tqdm

        indices = tqdm(indices)
        return self.sample_loop(
            *args,
            sampler=partial(
                self.p_sample_guidance,
                postprocess_fn=postprocess_fn,
                channel_last=channel_last,
                p_bar=indices,
                classifier_free_guidance_weight=classifier_free_guidance_weight,
            ),
            **kwargs,
        )
