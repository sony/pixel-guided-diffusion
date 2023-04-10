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
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import blobfile as bf
from torchvision import transforms

# from guided_diffusion.guided_diffusion.image_datasets import _list_image_files_recursively
from ddpm_segmentation.src.data_util import get_palette
from ddpm_segmentation.src.utils import to_labels


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class FeatureNpyDataset(Dataset):
    """
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    """

    def __init__(self, X_path: list, y_path: list, data_length: int, interval: int):
        self.X_path = X_path
        self.y_path = y_path
        self.interval = interval
        self.data_length = data_length

    def __getitem__(self, index):
        import time

        start = time.perf_counter()
        data_ind, pixel_ind = divmod(index, self.interval)

        start = time.perf_counter()
        X_data = np.load(self.X_path[data_ind])
        y_data = np.load(self.y_path[data_ind])

        start = time.perf_counter()
        X_pixel = X_data[pixel_ind]
        y_pixel = y_data[pixel_ind]

        return torch.from_numpy(X_pixel), torch.from_numpy(np.array(y_pixel))

    def __len__(self):
        return self.data_length


class ImageDataset(Dataset):
    """
    :param data_dir: path to a folder with images and their annotations.
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    """

    def __init__(
        self,
        img_path: str,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.image_paths = img_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        imgname = os.path.splitext(os.path.basename(image_path))[0]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        assert (
            pil_image.size[0] == pil_image.size[1]
        ), f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)

        return imgname, tensor_image


class PairImageDataset(Dataset):
    """
    :param data_dir: path to a folder with images and their annotations.
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    """

    def __init__(
        self,
        img_path: str,
        img_path2: str,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.image_paths = img_path
        self.image_paths2 = img_path2

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        image_path2 = self.image_paths2[idx]
        imgname = os.path.splitext(os.path.basename(image_path))[0]
        pil_image = Image.open(image_path)
        pil_image2 = Image.open(image_path2)
        pil_image = pil_image.convert("RGB")
        pil_image2 = pil_image2.convert("RGB")
        assert (
            pil_image.size[0] == pil_image.size[1]
        ), f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)
        tensor_image2 = self.transform(pil_image2)

        return imgname, tensor_image, tensor_image2


class ImageReferenceLabelDataset(Dataset):
    """
    :param data_dir: path to a folder with images and their annotations.
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    """

    def __init__(
        self,
        image_path: str,
        reference_path: str,
        resolution: int,
        category: str,
        transform=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = image_path
        self.reference_paths = reference_path
        self.palette = get_palette(category)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        reference_path = self.reference_paths[idx]
        imagename = os.path.basename(image_path).split("_")[0]

        reference_path = self.reference_paths[idx]
        referencename = os.path.basename(reference_path).split("_")[0]

        dirname = os.path.dirname(image_path)
        ref_dirname = os.path.dirname(reference_path)
        # Load an image
        image_mask_path = os.path.join(dirname, (imagename + "_raw.png"))
        reference_mask_path = os.path.join(
            ref_dirname, (referencename + "_raw.png"))

        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        assert (
            pil_image.size[0] == pil_image.size[1]
        ), f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"
        tensor_image = self.transform(pil_image)

        pil_ref = Image.open(reference_path)
        pil_ref = pil_ref.convert("RGB")
        tensor_ref = self.transform(pil_ref)

        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        mask_tensor = []
        for label in [image_mask_path, reference_mask_path]:
            label_np = cv2.imread(label)
            mask = to_labels(label_np, self.palette).astype("uint8")
            mask = cv2.resize(
                mask,
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_NEAREST,
            )
            tensor_label = torch.from_numpy(mask)
            mask_tensor.append(tensor_label)

        return (
            imagename,
            referencename,
            tensor_image,
            tensor_ref,
            mask_tensor[0],
            mask_tensor[1],
        )


class ImageEditLabelDataset(Dataset):
    """
    :param data_dir: path to a folder with images and their annotations.
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    """

    def __init__(
        self,
        edit_mask_paths: str,
        resolution: int,
        category: str,
        transform=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.edit_mask_paths = edit_mask_paths
        self.palette = get_palette(category)

    def _to_labels(self, masks, palette):
        results = np.zeros((256, 256), dtype=np.int32).reshape(-1)
        mask_flat = masks.reshape(-1, 3)  # 256*256, 3
        label = 0
        num = 0
        for color in np.array(palette).reshape(-1, 3):
            idxs = np.where((mask_flat == color).all(-1))
            results[idxs] = label
            label += 1
        return results.reshape(256, 256)

    def __len__(self):
        return len(self.edit_mask_paths)

    def __getitem__(self, idx):
        self.edit_mask_path = self.edit_mask_paths[idx]
        self.editname = os.path.splitext(
            os.path.basename(self.edit_mask_path))[0]
        self.imgname = os.path.basename(self.edit_mask_path).split("_")[0]
        self.dirname = os.path.dirname(self.edit_mask_path)
        # Load an image
        image_path = os.path.join(self.dirname, (self.imgname + "_img.png"))
        self.org_mask_path = os.path.join(
            self.dirname, (self.imgname + "_raw.png"))

        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        assert (
            pil_image.size[0] == pil_image.size[1]
        ), f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)

        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        mask_tensor = []
        for label in [self.org_mask_path, self.edit_mask_path]:
            label_np = cv2.imread(label)
            mask = self._to_labels(
                label_np, self.palette).astype("uint8")
            mask = cv2.resize(
                mask,
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_NEAREST,
            )
            tensor_label = torch.from_numpy(mask)
            mask_tensor.append(tensor_label)

        return self.editname, tensor_image, mask_tensor[0], mask_tensor[1]

    def roi_save(self, roi_np, save_img_dir):
        label = cv2.imread(self.org_mask_path)
        edit_label = cv2.imread(self.edit_mask_path)

        roi_np = roi_np.reshape(-1)

        label_alpha = cv2.cvtColor(label, cv2.COLOR_BGR2BGRA).reshape(-1, 4)
        edit_alpha = cv2.cvtColor(
            edit_label, cv2.COLOR_BGR2BGRA).reshape(-1, 4)

        label_alpha[np.where(roi_np == 0)[0], -1] = 60
        edit_alpha[np.where(roi_np == 0)[0], -1] = 60
        print("roi saved to ", save_img_dir)

        cv2.imwrite(
            os.path.join(save_img_dir, (self.editname + "_label_roi.png")),
            label_alpha.reshape(256, 256, 4),
        )
        cv2.imwrite(
            os.path.join(
                save_img_dir, (self.editname + "_label_edit_roi.png")),
            edit_alpha.reshape(256, 256, 4),
        )


def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type in ["ddpm", "guidance_ddpm"]:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type in ['mae', 'swav', 'swav_w2', 'deeplab']:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform


class FeatureDataset(Dataset):
    ''' 
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''

    def __init__(
        self,
        X_data: torch.Tensor,
        y_data: torch.Tensor
    ):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class ImageLabelDataset(Dataset):
    ''' 
    :param data_dir: path to a folder with images and their annotations. 
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images=-1,
        transform=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)
        self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        assert pil_image.size[0] == pil_image.size[1], \
            f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(
            label, (self.resolution,
                    self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label


class InMemoryImageLabelDataset(Dataset):
    ''' 

    Same as ImageLabelDataset but images and labels are already loaded into RAM.
    It handles DDPM/GAN-produced datasets and is used to train DeepLabV3. 

    :param images: np.array of image samples [num_images, H, W, 3].
    :param labels: np.array of correspoding masks [num_images, H, W].
    :param resolution: image and mask output resolusion.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''

    def __init__(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            resolution=256,
            transform=None
    ):
        super().__init__()
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        assert image.size[0] == image.size[1], \
            f"Only square images are supported: ({image.size[0]}, {image.size[1]})"

        tensor_image = self.transform(image)
        label = self.labels[idx]
        label = cv2.resize(
            label, (self.resolution,
                    self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label
