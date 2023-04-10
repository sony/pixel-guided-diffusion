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

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
from distutils.util import strtobool
from glob import glob
from posixpath import basename
from unittest import result

import cv2
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ddpm_segmentation.guided_diffusion.guided_diffusion import logger
from ddpm_segmentation.guided_diffusion.guided_diffusion.script_util import (
    add_dict_to_argparser, classifier_and_diffusion_defaults,
    model_and_diffusion_defaults)
from ddpm_segmentation.src.data_util import get_palette
from ddpm_segmentation.src.utils import colorize_mask

from src.datasets import ImageEditLabelDataset, make_transform
from src.feature_extractors import create_feature_extractor
from src.pixel_classifier import (extract_mlp_features, load_ensemble,
                                  load_multiple_ensemble, pixel_classifier,
                                  predict_labels, predict_mean_logits)
from src.segmentation_guided_diffusion import (collect_features,
                                               collect_features_wo_hook)
from src.utils import dev, extract_roi, torch_fix_seed, ensure_dir


def load_data(opts):
    if os.path.isdir(opts["edit_map_path"]):
        logger.log("## Multiple Input")
        opts["edit_map_path"] = sorted(
            glob(os.path.join(opts["edit_map_path"], "*_edited.png"))
        )

    else:  # single images
        logger.log("## Single Input")
        opts["edit_map_path"] = [opts["edit_map_path"]]


def evaluate(samples, label_edit, save_dir, editname, roi):
    criterion = nn.CrossEntropyLoss(reduction=opts["loss_reduction"])
    original_dir = opts["exp_dir"]
    opts_eval = opts.copy()
    opts_eval["exp_dir"] = opts["evaluate_weight_path"]
    opts_eval["dim"] = [256, 256, 8448]
    opts_eval["model_type"] = "ddpm"
    opts_eval["diffusion_steps"] = 1000
    opts_eval["timestep_respacing"] = "1000"
    feature_extractor = create_feature_extractor(**opts_eval)
    models = load_ensemble(opts_eval, device=dev())

    if "share_noise" in opts_eval and opts_eval["share_noise"]:
        rnd_gen = th.Generator(device=dev()).manual_seed(opts_eval["seed"])
        noise = th.randn(
            1,
            3,
            opts_eval["image_size"],
            opts_eval["image_size"],
            generator=rnd_gen,
            device=dev(),
        )
    else:
        noise = None

    for ind, sample in enumerate(samples):
        palette = get_palette(opts_eval["category"])

        img = sample[None].to("cuda")
        features = feature_extractor(img, noise=noise)
        features = collect_features(opts_eval, features)

        x = features.view(opts_eval["dim"][-1], -1).permute(1, 0)

        roi_flatten = roi.view(-1)
        y = label_edit.view(-1)
        y = y[roi_flatten > 0].view(-1)

        pred, uncertainty_score = predict_labels(
            models, x, size=opts_eval["dim"][:-1])
        mean_logits = predict_mean_logits(
            models, x, size=opts_eval["dim"][:-1])
        mean_logits = mean_logits.view(-1, opts_eval["number_class"])

        pred = th.argmax(mean_logits, -1)
        mask = colorize_mask(
            pred.cpu().data.numpy().reshape(256, 256), palette)
        pred = pred[roi_flatten > 0].view(-1)
        mean_logits = mean_logits[roi_flatten > 0, :]
        accuracy = sum(pred == y) / len(y)
        loss = criterion(mean_logits, y)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(1, 2, 0)
        sample = sample.contiguous()
        sample = th.squeeze(sample)

        edited_bgr = cv2.cvtColor(np.squeeze(
            sample.cpu().numpy()), cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                result_dir,
                (f"batch_{ind}_loss_{loss.item()}_acc_{accuracy}_img" +
                 editname + ".png"),
            ),
            edited_bgr,
        )
        cv2.imwrite(
            os.path.join(
                result_dir,
                (
                    f"batch_{ind}_loss_{loss.item()}_acc_{accuracy}_mask"
                    + editname
                    + ".png"
                ),
            ),
            mask,
        )


def main():
    runtime_dict = dict()
    diffusion_process = "ddim" if args.use_ddim else "ddpm"
    savedir = opts["exp_dir"]

    logger.configure(dir=savedir)
    logger.log(f"Diffusion Process: {diffusion_process}...")

    logger.log("Loading pretrained classifiers...")
    if args.seg_cond:
        classifiers_dict = load_multiple_ensemble(
            opts,
            device="cuda",
            middle_step=args.max_steps,
            interval=args.cond_fn_interval,
        )

    else:
        classifiers_dict = None
    logger.log("Sampling noise for feature extraction...")

    if opts["share_noise"]:
        rnd_gen = th.Generator(device="cuda").manual_seed(args.seed)
        seg_noise = th.randn(
            1,
            3,
            opts["image_size"],
            opts["image_size"],
            generator=rnd_gen,
            device="cuda",
        )
    else:
        seg_noise = None

    model_kwargs = {}

    load_data(opts)

    dataset = ImageEditLabelDataset(
        edit_mask_paths=opts["edit_map_path"],
        resolution=opts["image_size"],
        category=opts["category"],
        transform=make_transform(opts["model_type"], opts["image_size"]),
    )

    with open(opts["roi_cls_file"], "r") as f:
        edit_cls_dict = json.load(f)

    logger.log(f"## Dataset Length: {len(dataset)}")

    for editname, img, label, label_edit in dataset:
        logger.log("editname: ", editname)
        img = img.to("cuda")
        if len(img.size()) == 3:
            img = img.unsqueeze(0)

        cls_ids = edit_cls_dict[editname]
        dilate_roi, pixel_num = extract_roi(opts, cls_ids, label, label_edit)

        # Set t_0, guidance scale
        if pixel_num["whole"] > 5000:  # Large Part's Manipulation
            t_0, seg_scale = 750, 40
        else:  # Small Part's Manipulation
            t_0, seg_scale = 500, 100
        if args.segmentation_scale:
            seg_scale = args.segmentation_scale
        if args.t0:
            t_0 = args.t0
        opts["start_step"] = t_0
        opts["segmentation_scale"] = seg_scale * args.seg_scale
        if args.timestep_respacing == "1000":
            opts["timestep_respacing"] = str(t_0)
        else:
            opts["timestep_respacing"] = args.timestep_respacing

        logger.log("Creating feature extractor...")
        feature_extractor = create_feature_extractor(**opts)
        model, diffusion = feature_extractor.model, feature_extractor.diffusion
        use_steps = diffusion.timestep_map
        logger.log(f"## Hyper parameters: t_0: {t_0}, scale: {seg_scale}")

        reverse_sample_fn = diffusion.ddim_reverse_sample_loop
        if args.seg_cond:
            sample_fn = (
                diffusion.p_sample_loop_guidance if not args.use_ddim else diffusion.ddim_sample_loop_guidance
            )

        else:
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

        save_img_dir = os.path.join(opts["exp_dir"], editname, "f_{}_r_{}_s_{}".format(
            opts["timestep_respacing"], opts["timestep_respacing"], seg_scale))

        dataset.roi_save(dilate_roi, save_img_dir)

        # make saveral save directory
        global result_dir
        result_dir = os.path.join(save_img_dir, "edited")
        ensure_dir(result_dir)


        with open(os.path.join(save_img_dir, "expriment_condition.json"), "w") as f:
            json.dump(opts, f, indent=4)

        dilate_roi = th.from_numpy(dilate_roi).to("cuda")
        label_edit = label_edit.to("cuda")

        @th.no_grad()
        def postprocess_fn(out, t, roi):
            background_stage_t = diffusion.q_sample(img, t[0])
            background_stage_t = th.tile(
                background_stage_t, dims=(args.batch_size, 1, 1, 1)
            )
            out["sample"] = out["sample"] * roi + \
                background_stage_t * (1 - roi)

            return out

        import time
        th.cuda.synchronize()
        start_time = time.time()

        logger.log("Generating Noise by Deterministic ...")
        reverse_output = reverse_sample_fn(
            model=model,
            clip_denoised=args.clip_denoised,
            x=img,
            device="cuda"
        )

        X_T = reverse_output["sample"].clone()

        if args.batch_size > 1:
            X_T = th.tile(X_T, (args.batch_size, 1, 1, 1))
            label_edit = th.tile(th.unsqueeze(
                label_edit, 0), (args.batch_size, 1, 1))

        model_kwargs["y"] = label_edit
        model_kwargs["roi"] = dilate_roi
        postprocess_fn_ = postprocess_fn if args.postprocess else None

        logger.log("sampling...")
        if args.seg_cond:
            output = sample_fn(
                opts=opts,
                model=model,
                classifiers_dict=classifiers_dict,
                cond_fn=None,
                feature_extractor=feature_extractor,
                save_dir=None,
                shape=(args.batch_size, 3,
                       opts["image_size"], opts["image_size"]),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                noise=X_T,
                device="cuda",
                progress=True,
                postprocess_fn=postprocess_fn_,
                roi=dilate_roi,
            )
        else:
            output = sample_fn(
                model=model,
                shape=(args.batch_size, 3,
                       opts["image_size"], opts["image_size"]),
                noise=X_T,
                clip_denoised=args.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
                device="cuda",
            )

        th.cuda.synchronize()
        editing_time = time.time() - start_time
        logger.log("editing time,", editing_time)
        runtime_dict[editname] = editing_time
        info = {"runtime": editing_time,
                "t_0": t_0,
                "scale": opts["segmentation_scale"]}
        with open(os.path.join(save_img_dir, editname + "_runtime.json"), "w") as f:
            json.dump(info, f)

        evaluate(
            output,
            th.chunk(label_edit, opts["batch_size"])[0],
            result_dir,
            editname,
            dilate_roi,
        )

        del img, X_T, dilate_roi, label, label_edit
        th.cuda.empty_cache()
    with open(os.path.join(opts["exp_dir"], "runtime.json"), "w") as f:
        json.dump(runtime_dict, f)


def create_argparser():
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument("--exp", type=str)
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--segmentation_scale", type=float, default=0)
    parser.add_argument("--edit_map_path", type=str)
    parser.add_argument(
        "--loss_reduction", type=str, choices=["mean", "sum"], default="mean"
    )
    parser.add_argument("--seg_cond", type=strtobool, default=True, help="whether perform pixel-wise guidance or simple ddim reconstruction")
    parser.add_argument("--seg_scale", type=float, default=1.0)
    parser.add_argument("--clip_denoised", type=strtobool, default=True)
    parser.add_argument("--roi_cls_file", type=str)
    parser.add_argument("--use_ddim", type=strtobool)
    parser.add_argument("--max_steps", type=int, default=750)
    parser.add_argument("--t0", type=int, default=0)
    parser.add_argument("--postprocess", type=strtobool, default=True)
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--analyze", type=strtobool, default=False)
    parser.add_argument("--cond_fn_interval", type=int, default=0)
    parser.add_argument("--evaluate_weight_path", type=str)
    parser.add_argument("--steps", type=int, default=[50, 150, 250])

    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    opts = json.load(open(args.exp, "r"))
    opts.update(vars(args))
    opts["image_size"] = opts["dim"][0]
    torch_fix_seed()

    main()
