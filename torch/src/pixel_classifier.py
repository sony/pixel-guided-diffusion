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
from pickletools import uint8
import cv2
import shutil
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from glob import glob
from tqdm import tqdm

from torchvision.models.feature_extraction import create_feature_extractor
from torch.distributions import Categorical
from ddpm_segmentation.src.utils import colorize_mask, oht_to_scalar, to_labels
from ddpm_segmentation.src.data_util import get_palette, get_class_names
from PIL import Image


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim, extract_feature=False):
        super(pixel_classifier, self).__init__()
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class),
            )

    def init_weights(self, init_type="normal", gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)


def predict_labels(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    # from scipy.stats import entropy as en

    mean_seg = None
    all_seg = []
    all_entropy, all_entropy_ = [], []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            if not next(models[MODEL_NUMBER].parameters()).is_cuda:
                models[MODEL_NUMBER].cuda()
            preds = models[MODEL_NUMBER](features.cuda())
            entropy = Categorical(logits=preds).entropy()

            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        # print("size mean seg: ", mean_seg.size())
        # print("mean seg 0: ", mean_seg[0, :])

        mean_seg_np = mean_seg.cpu().data.numpy()
        full_entropy = Categorical(mean_seg).entropy()
        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][-int(js.shape[0] / 10):].mean()  # 10%

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k


def predict_labels_wo_entropy(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)

    mean_seg = None
    all_seg = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            if not next(models[MODEL_NUMBER].parameters()).is_cuda:
                models[MODEL_NUMBER].cuda()
            preds = models[MODEL_NUMBER](features.cuda())
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)
        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final


def predict_mean_logits(models, features, size):
    # meaning of multiple logits
    batch_size = 1
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    if len(features.size()) == 3:
        batch_size = features.size()[0]
        features = features.view(-1, size[-1])

    seg_mode_ensemble = []
    # softmax_f = nn.Softmax(dim=1)
    for MODEL_NUMBER in range(len(models)):
        model = models[MODEL_NUMBER].to("cuda")
        preds = model(features.cuda())
        preds = preds.view(batch_size, -1, preds.size()[-1])
        seg_mode_ensemble.append(preds.unsqueeze(0))
        del model
        torch.cuda.empty_cache()
    seg_mode_mean = torch.mean(torch.cat(seg_mode_ensemble, 0), 0)

    return seg_mode_mean


def extract_mlp_features(opts, models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)

    model = models[0].to("cuda")
    if opts["calculation_type"] == "after_relu":
        feature_extractor = create_feature_extractor(
            model, ["layers.1", "layers.4", "layers.6"]
        )
    elif opts["calculation_type"] == "after_bn":
        # after batch normalization
        feature_extractor = create_feature_extractor(
            model, ["layers.2", "layers.5", "layers.6"]
        )
    else:
        feature_extractor = create_feature_extractor(
            model, ["layers.0", "layers.3", "layers.6"]
        )
    middle_features = feature_extractor(features)
    del model
    torch.cuda.empty_cache()

    return middle_features


def save_predictions(args, image_paths, preds, gts, imgs):
    palette = get_palette(args["category"])
    os.makedirs(os.path.join(args["exp_dir"], "predictions"), exist_ok=True)
    os.makedirs(os.path.join(args["exp_dir"], "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args["exp_dir"], "raw_images"), exist_ok=True)
    os.makedirs(os.path.join(args["exp_dir"], "noise_images"), exist_ok=True)

    for i, (img, pred, gt) in enumerate(zip(imgs, preds, gts)):
        len_output = len(img) + 2
        save_format = (
            np.ones(
                (pred.shape[0], pred.shape[1] *
                 len_output + 30 * (len_output - 1), 3),
                dtype=np.uint8,
            )
            * 255
        )
        filename = image_paths[i].split("/")[-1].split(".")[0]
        np.save(os.path.join(args["exp_dir"],
                "predictions", filename + ".npy"), pred)

        mask_gt = colorize_mask(gt, palette)  # gt:256,256
        mask = colorize_mask(pred, palette)  # pred:256,256
        label_convert = to_labels(
            np.expand_dims(mask, 0), palette
        )  # label_convert:256,256
        shutil.copy(
            image_paths[i],
            os.path.join(args["exp_dir"], "raw_images", filename + ".png"),
        )
        cv2.imwrite(
            os.path.join(args["exp_dir"], "predictions",
                         filename + ".png"), mask
        )

        if len(img) == 1:
            img = img[0]
            img = ((img + 1) / 2) * 255  # Decompose Preprocessing
            img_ = cv2.cvtColor(np.squeeze(img).transpose(
                1, 2, 0), cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(args["exp_dir"], "noise_images",
                             filename + ".png"), img_
            )

            save_format[:, : pred.shape[1], :] = np.squeeze(
                img).transpose(1, 2, 0)
            j = 1

        else:
            img = np.stack(img)
            img = ((img + 1) / 2) * 255  # Decompose Preprocessing
            for j, im in enumerate(img):
                if j == 0:
                    save_format[:, : pred.shape[1], :] = np.squeeze(im).transpose(
                        1, 2, 0
                    )
                else:
                    save_format[
                        :,
                        pred.shape[1] * j + 30 * j: pred.shape[1] * (j + 1) + 30 * j,
                        :,
                    ] = np.squeeze(im).transpose(1, 2, 0)
            j += 1

        save_format[
            :, pred.shape[1] * j + 30 * j: pred.shape[1] * (j + 1) + 30 * j, :
        ] = mask_gt
        save_format[:, pred.shape[1] * (j + 1) + 30 * (j + 1):, :] = mask

        Image.fromarray(save_format).save(
            os.path.join(args["exp_dir"], "visualizations", filename + ".jpg")
        )


def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args["category"])

    ids = range(args["number_class"])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args["ignore_label"]:
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()

    ious = []
    for target_num in ids:
        if target_num == args["ignore_label"]:
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean()


def load_ensemble(args, device="cpu"):
    models = []
    for i in range(args["model_num"]):
        model_path = os.path.join(args["exp_dir"], f"model_{i}.pth")
        state_dict = torch.load(model_path)["model_state_dict"]
        model = nn.DataParallel(pixel_classifier(
            args["number_class"], args["dim"][-1]))
        model.load_state_dict(state_dict)
        model = model.module.to(device)
        models.append(model.eval())
    return models


def load_multiple_ensemble(args, device="cpu", middle_step=0, interval=0):
    models_dict = {}
    print(f"Load {middle_step} models...")

    if interval:
        download_step_list = [i for i in range(
            middle_step) if i % interval == 0]
        print(download_step_list)

    else:
        download_step_list = [i for i in range(middle_step)]

    for step in tqdm(download_step_list):

        models = []
        suffix = f"{step}_"
        suffix += "_".join([str(step) for step in args["blocks"]])
        for i in range(args["model_num"]):
            model_path = os.path.join(
                args["mlp_model_path"], suffix, f"model_{i}.pth")
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]

            from collections import OrderedDict

            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if "module" in k:
                    k = k.replace("module.", "")
                new_state_dict[k] = v

            model = pixel_classifier(args["number_class"], args["dim"][-1])
            model.load_state_dict(new_state_dict, strict=True)
            models.append(model.eval())  # device:cpu

        for c in models:
            for i in c.parameters():
                i.requires_grad = False

        models_dict[step] = models

    return models_dict
