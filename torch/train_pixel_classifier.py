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

# This code is revised training code for mutliple steps based on "train_interpreter.py" from (https://github.com/yandex-research/ddpm-segmentation)

import argparse
import gc
import json
import os
from distutils.util import strtobool

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm_segmentation.guided_diffusion.guided_diffusion.script_util import (
    add_dict_to_argparser, model_and_diffusion_defaults)
from ddpm_segmentation.src.utils import multi_acc, setup_seed

from src.datasets import FeatureDataset, ImageLabelDataset, make_transform
from src.feature_extractors import create_feature_extractor
from src.pixel_classifier import (compute_iou, load_ensemble, pixel_classifier,
                                  predict_labels, predict_mean_logits,
                                  save_predictions)
from src.segmentation_guided_diffusion import collect_features
from src.utils import dev


def prepare_data(args):
    feature_extractor = create_feature_extractor(**args)

    print(f"Preparing the train set for {args['category']}...")
    dataset = ImageLabelDataset(
        data_dir=args["training_path"],
        resolution=args["image_size"],
        num_images=args["training_number"],
        transform=make_transform(args["model_type"], args["image_size"]),
    )

    X = torch.zeros((len(dataset), *args["dim"][::-1]), dtype=torch.float)
    y = torch.zeros((len(dataset), *args["dim"][:-1]), dtype=torch.uint8)

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

    for row, (img, label) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        X[row] = collect_features(args, features).cpu()  # (2800, 256, 256)

        for target in range(args["number_class"]):
            if target == args["ignore_label"]:
                continue
            if 0 < (label == target).sum() < 20:
                print(
                    f"Delete small annotation from image {dataset.image_paths[row]} | label {target}"
                )
                label[label == target] = args["ignore_label"]
        y[row] = label

    d = X.shape[1]
    print(f"Total dimension {d}")

    X = X.permute(1, 0, 2, 3).reshape(d, -1).permute(1, 0)  # (2800, 33000000)
    y = y.flatten()  # (3000000)

    return X, y


def evaluation(args, models):
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageLabelDataset(
        data_dir=args["testing_path"],
        resolution=args["image_size"],
        num_images=args["testing_number"],
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
    for img, label in tqdm(dataset):

        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args["dim"][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args["dim"][:-1])
        mean_logits = predict_mean_logits(models, x, size=args["dim"][:-1])

        gts.append(label.numpy())
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item())

    save_predictions(args, dataset.image_paths, preds,
                     gts, feature_extractor.input)
    miou = compute_iou(args, preds, gts)
    print(f"Overall mIoU: ", miou)
    print(
        f"Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}")

    result_dict = {
        "mIOU": miou,
        "uncertainty": sum(uncertainty_scores) / len(uncertainty_scores),
    }
    num_step = args["steps"]
    with open(os.path.join(args["exp_dir"], f"{num_step}_steps.json"), "w") as f:
        json.dump(result_dict, f, indent=4)


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
    print("training start")

    features, labels = prepare_data(args)
    print("prepared..............")
    train_data = FeatureDataset(features, labels)

    print(
        f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********"
    )
    print(
        f" *********************** Current number data {len(features)} ***********************"
    )

    train_loader = DataLoader(
        dataset=train_data, batch_size=args["batch_size"], shuffle=True, drop_last=True
    )

    print(
        " *********************** Current dataloader length "
        + str(len(train_loader))
        + " ***********************"
    )

    for MODEL_NUMBER in range(args["start_model_num"], args["model_num"], 1):

        gc.collect()
        classifier = pixel_classifier(
            numpy_class=(args["number_class"]), dim=args["dim"][-1]
        )
        classifier.init_weights()

        classifier = nn.DataParallel(classifier).to(dev())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in range(100):
            print("Epoch: ", epoch)
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                y_batch = y_batch.type(torch.long)

                X_batch = X_batch[y_batch != args["ignore_label"]]
                y_batch = y_batch[y_batch != args["ignore_label"]]

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)
                acc = multi_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print(
                        "Epoch : ",
                        str(epoch),
                        "iteration",
                        iteration,
                        "loss",
                        loss.item(),
                        "acc",
                        acc,
                    )

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print(
                            "*************** Break, Total iters,",
                            iteration,
                            ", at epoch",
                            str(epoch),
                            "***************",
                        )
                        break

            if stop_sign == 1:
                break

        model_path = os.path.join(
            args["exp_dir"], "model_" + str(MODEL_NUMBER) + ".pth"
        )
        MODEL_NUMBER += 1
        print("save to:", model_path)
        torch.save({"model_state_dict": classifier.state_dict()}, model_path)


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

    if not all(pretrained):
        # train all remaining models
        opts["start_model_num"] = sum(pretrained)
        train(opts)

        print("Loading pretrained models...")
        models = load_ensemble(opts, device=dev())
        evaluation(opts, models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument("--exp", type=str)
    parser.add_argument("--middle_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 150, 250])
    parser.add_argument("--seg_cond", type=strtobool, default=False)
    # parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()
    setup_seed(args.seed)
    # Load the experiment config
    opts = json.load(open(args.exp, "r"))
    opts.update(vars(args))
    opts["image_size"] = opts["dim"][0]

    main()
