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

from typing import List

import numpy as np
import torch as th
from torch import nn
from tqdm import tqdm

from ddpm_segmentation.guided_diffusion.guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, _extract_into_tensor)
from ddpm_segmentation.guided_diffusion.guided_diffusion.respace import \
    _WrappedModel
from ddpm_segmentation.src.data_util import get_palette
from ddpm_segmentation.src.utils import colorize_mask
from src.pixel_classifier import (extract_mlp_features, load_multiple_ensemble,
                                  predict_labels, predict_labels_wo_entropy,
                                  predict_mean_logits)


def collect_features(args, activations: List[th.Tensor], sample_idx=0):
    """Upsample activations and concatenate them to form a feature tensor"""
    assert all([isinstance(acts, th.Tensor) for acts in activations])
    size = tuple(args["dim"][:-1])
    resized_activations = []
    for feats in activations:

        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"])
        resized_activations.append(feats[0])

    return th.cat(resized_activations, dim=0)


def collect_features_wo_hook(args, activations: List[th.Tensor], sample_idx=0):
    """Upsample multiple activations gathered by w/o hook function and concatenate them to form a feature tensor"""
    assert all([isinstance(acts, th.Tensor) for acts in activations])
    size = tuple(args["dim"][:-1])
    resized_activations = []

    for feats in activations:
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"])

        resized_activations.append(feats)
    return th.cat(resized_activations, dim=1)


class SegmentationGaussianDiffusion(GaussianDiffusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_timesteps = int(self.betas.shape[0])
        self.init_noise = None

    def resampling_mean(self, p_mean_var):

        new_mean = (
            p_mean_var["mean"].float()
            + p_mean_var["variance"] * p_mean_var["gradient"].float()
        )
        return new_mean

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                    as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                - 'mean': the model mean output.
                - 'variance': the model variance output.
                - 'log_variance': the log of 'variance'.
                - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]

        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(
                model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_mean_variance_guidance(
        self,
        model,
        opts,
        classifiers_dict,
        x,
        t,
        cond_fn=None,
        new_ts=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        statistics_file=None,
        statistics_dict=None,
        analyze=False,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        criterion = nn.CrossEntropyLoss(reduction=opts["loss_reduction"])

        int_t = int(t[0]) if len(t) > 1 else int(t)
        new_t = model.timestep_map[int_t]

        classifiers = classifiers_dict[int(new_t)]

        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            model_output, activations_list = model(
                x_in, self._scale_timesteps(t), **model_kwargs
            )

            features = collect_features_wo_hook(opts, activations_list)
            x_middle = (
                features.view(opts["batch_size"], opts["dim"][-1], -1)
                .permute(0, 2, 1)
                .to(th.float32)
            )  # (65536, 2816)
            # Flatten to (N*H*W, opts["dim"][-1])
            y = model_kwargs["y"].view(opts["batch_size"], -1)
            roi_flatten = model_kwargs["roi"].view(-1)
            x_middle = x_middle[:, roi_flatten > 0, :]
            y = y[:, roi_flatten > 0].view(-1)
            # TODO: ROI Features only INPUT
            mean_logits = predict_mean_logits(
                classifiers, x_middle, size=opts["dim"])
            mean_logits = mean_logits.view(-1, opts["number_class"])

            pred = th.argmax(mean_logits, -1).view(-1)
            accuracy = sum(pred == y) / len(y)
            loss = criterion(mean_logits, y) * opts["batch_size"]
            # recording loss to dictionary

            gradient = th.autograd.grad(loss, x_in)[0]
            gradient_scaled = -opts["segmentation_scale"] * gradient
            del gradient

        if analyze:
            statistics_dict["cross_entropy"].update({int(int_t): loss.item()})
            statistics_dict["accuracy"].update({int(int_t): float(accuracy)})
            with open(statistics_file, "w") as f:
                json.dump(statistics_dict, f, indent=4)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:

            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(
                model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "gradient": gradient_scaled,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample_guidance(
        self,
        opts,
        model,
        classifiers_dict,
        save_dir,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        analyze=False,
        roi=None,
        statistics_file=None,
        statistics_dict=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """

        out = self.p_mean_variance_guidance(
            model,
            opts,
            classifiers_dict,
            x,
            t,
            cond_fn=None,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            statistics_file=statistics_file,
            statistics_dict=statistics_dict,
            analyze=analyze,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        out["mean"] = self.resampling_mean(out)

        sample = out["mean"] + nonzero_mask * \
            th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_guidance(
        self,
        opts,
        model,
        classifiers_dict,
        feature_extractor,
        save_dir,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        cond_fn_interval=0,
        model_kwargs=None,
        device=None,
        progress=True,
        postprocess_fn=None,
        roi=None,
        analyze=False,
        statistics_file=False,
        statistics_dict=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        before_roi, diff_roi, grad_roi, var_roi = [], [], [], []
        for i, sample in enumerate(
            self.p_sample_loop_progressive_guidance(
                opts,
                model,
                classifiers_dict,
                feature_extractor,
                save_dir,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                cond_fn_interval=cond_fn_interval,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                postprocess_fn=postprocess_fn,
                roi=roi,
                analyze=analyze,
                statistics_file=statistics_file,
                statistics_dict=statistics_dict,
            )
        ):
            final = sample

        return final["sample"]

    def p_sample_loop_progressive_guidance(
        self,
        opts,
        model,
        classifiers_dict,
        feature_extractor,
        save_dir,
        shape,
        start_step=False,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        cond_fn_interval=None,
        model_kwargs=None,
        device=None,
        progress=False,
        postprocess_fn=None,
        roi=None,
        analyze=False,
        statistics_file=False,
        statistics_dict=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        self.init_noise = th.clone(img).cpu()
        indices = list(range(self.num_timesteps))[::-1]
        for i in tqdm(indices):
            t = th.tensor([i] * shape[0], device=device)

            with th.no_grad():
                # out = self.p_sample(
                out = self.p_sample_guidance(
                    opts,
                    model,
                    classifiers_dict,
                    save_dir,
                    img,
                    t,
                    cond_fn=cond_fn,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    analyze=analyze,
                    roi=roi,
                    statistics_file=statistics_file,
                    statistics_dict=statistics_dict,
                )
                if postprocess_fn is not None:
                    out = postprocess_fn(out, t, roi)

                yield out
                img = out["sample"]

    def resampling_out(self, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        gradient = p_mean_var["gradient"]

        eps = eps - (1 - alpha_bar).sqrt() * gradient

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def ddim_sample_guidance(
        self,
        opts,
        model,
        classifiers_dict,
        save_dir,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        analyze=False,
        roi=None,
        statistics_file=None,
        statistics_dict=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance_guidance(
            model,
            opts,
            classifiers_dict,
            x,
            t,
            cond_fn=None,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            statistics_file=statistics_file,
            statistics_dict=statistics_dict,
            analyze=analyze,
        )
        out = self.resampling_out(out, x, t, model_kwargs=model_kwargs)

        if cond_fn is not None:
            out = self.condition_score(
                cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(
            self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop_guidance(
        self,
        opts,
        model,
        classifiers_dict,
        feature_extractor,
        save_dir,
        shape,
        start_step=False,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        eta=0.0,
        postprocess_fn=None,
        roi=None,
        analyze=False,
        statistics_dict=False,
        statistics_file=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive_guidance(
            opts,
            model,
            classifiers_dict,
            feature_extractor,
            save_dir,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            start_step=start_step,
            progress=progress,
            eta=eta,
            postprocess_fn=postprocess_fn,
            roi=roi,
            analyze=analyze,
            statistics_dict=statistics_dict,
            statistics_file=statistics_file,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive_guidance(
        self,
        opts,
        model,
        classifiers_dict,
        feature_extractor,
        save_dir,
        shape,
        start_step=False,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        eta=0.0,
        postprocess_fn=None,
        roi=None,
        analyze=False,
        statistics_dict=False,
        statistics_file=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample_guidance(
                    opts,
                    model,
                    classifiers_dict,
                    save_dir,
                    img,
                    t,
                    cond_fn=cond_fn,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    analyze=analyze,
                    roi=roi,
                    statistics_file=statistics_file,
                    statistics_dict=statistics_dict,
                )

                if postprocess_fn is not None:
                    out = postprocess_fn(out, t, roi)

                if analyze:
                    int_t = int(t[0]) if len(t) > 1 else int(t)

                    statistics_dict["xt_max"].update(
                        {int(int_t): (float(th.max(out["sample"])))}
                    )
                    statistics_dict["xt_min"].update(
                        {int(int_t): (float(th.min(out["sample"])))}
                    )
                    with open(statistics_file, "w") as f:
                        json.dump(statistics_dict, f, indent=4)

                yield out
                img = out["sample"]

    def ddim_reverse_sample_loop(
        self,
        model,
        x,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        device=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        reference_roi = []
        T = []
        indices = list(range(self.num_timesteps))

        from tqdm.auto import tqdm

        indices = tqdm(indices)

        sample = x
        for i in indices:
            t = th.tensor([i] * len(sample), device=device)
            with th.no_grad():
                out = self.ddim_reverse_sample(
                    model,
                    sample,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

                sample = out["sample"]
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out["pred_xstart"])
                # [0, ..., T-1] ready to use
                T.append(t)

        return {
            #  xT "
            "sample": sample,
            # (1, ..., T)
            "sample_t": sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            "xstart_t": xstart_t,
            "ranking_roi": reference_roi,
            "T": T,
        }

    def segmentation_predict(self, opts, classifiers_dict, feature_extractor, x, t):

        int_t = int(t[0]) if len(t) > 1 else int(t)
        classifiers = classifiers_dict[int_t]
        softmax_f = nn.Softmax(dim=1)
        with th.enable_grad():
            x_in = x.detach()
            activations_list = feature_extractor(x_in, int_t)
            features = collect_features_wo_hook(opts, activations_list)

            # TODO: Analyze why float16 tensor was generated
            x = (
                features.view(opts["dim"][-1], -1).permute(1, 0).to(th.float32)
            )  # (65536, 2816)

            x_copy = x.detach()
            pred = predict_labels_wo_entropy(
                classifiers, x_copy, size=opts["dim"][:-1])
            mean_logits = predict_mean_logits(
                classifiers, x_copy, size=opts["dim"][:-1]
            )
            mean_logits_ = th.squeeze(mean_logits)
            softmax_output = softmax_f(mean_logits_)

            palette = get_palette(opts["category"])
            mask = colorize_mask(pred.numpy(), palette)
        return mask, pred, softmax_output


class SpacedSegmentationGuideDiffusion(SegmentationGaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(
            **kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        last_ind = self.timestep_map[-1] + \
            (self.timestep_map[-1] - self.timestep_map[-2])

        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

        try:
            self.alphas_cumprod_next = np.append(
                self.alphas_cumprod[1:], base_diffusion.alphas_cumprod[last_ind])
        except IndexError:
            self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def p_mean_variance_guidance(
        self, model, *args, **kwargs
    ):
        return super().p_mean_variance_guidance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t
