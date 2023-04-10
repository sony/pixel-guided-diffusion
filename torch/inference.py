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
import gc
import cv2
import json
import os
import torch
from distutils.util import strtobool
from glob import glob
from tqdm import tqdm
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from ddpm_segmentation.guided_diffusion.guided_diffusion.script_util import (
    add_dict_to_argparser, model_and_diffusion_defaults)
from ddpm_segmentation.src.data_util import get_palette
from ddpm_segmentation.src.utils import colorize_mask, multi_acc, setup_seed
from src.datasets import FeatureDataset, ImageDataset, make_transform
from src.feature_extractors import create_feature_extractor
from src.segmentation_guided_diffusion import collect_features
from src.pixel_classifier import (compute_iou, load_ensemble, pixel_classifier,
                                  predict_labels, predict_mean_logits,
                                  save_predictions)
from src.utils import dev



def evaluation(args, models):
    palette = get_palette(args["category"])
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageDataset(
        img_path=[args["img_path"]],
        transform=make_transform(args["model_type"], args["image_size"]),
    )

    if "share_noise" in args and args["share_noise"]:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args["seed"])
        noise = torch.randn(
            1,
            3,
            args["image_size"],
            args["image_size"],
            generator=rnd_gen,
            device=dev(),
        )
    else:
        noise = None

    print("Dataset Length: ", len(dataset))

    imgs, preds, gts, uncertainty_scores, logits = [], [], [], [], []
    for img_name, img in tqdm(dataset):

        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args["dim"][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(models, x, size=args["dim"][:-1])
        mean_logits = predict_mean_logits(models, x, size=args["dim"][:-1])
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item())
        mask = colorize_mask(pred.numpy(), palette)
        cv2.imwrite(
            os.path.join(args["mask_save_path"], (img_name.split("_")[0] + "_raw.png")),
            mask,
        )


def main():
    # Prepare the experiment folder
    if len(opts["steps"]) > 0:
        suffix = "_".join([str(step) for step in opts["steps"]])
        suffix += "_" + "_".join([str(step) for step in opts["blocks"]])
        opts["exp_dir"] = os.path.join(opts["exp_dir"], suffix)

    path = opts["exp_dir"]
    os.makedirs(path, exist_ok=True)
    print("Experiment folder: %s" % (path))
    os.system("cp %s %s" % (args.exp, opts["exp_dir"]))
    # Check whether all models in ensemble are trained
    pretrained = [
        os.path.exists(os.path.join(opts["exp_dir"], f"model_{i}.pth"))
        for i in range(opts["model_num"])
    ]

    print("Loading pretrained models...")
    models = load_ensemble(opts, device=dev())
    evaluation(opts, models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument("--exp", type=str)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 150, 250])
    parser.add_argument("--mask_save_path", type=str)

    args = parser.parse_args()
    setup_seed(args.seed)
    # Load the experiment config
    opts = json.load(open(args.exp, "r"))
    opts.update(vars(args))
    opts["image_size"] = opts["dim"][0]

    main()
