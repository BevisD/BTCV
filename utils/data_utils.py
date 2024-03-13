# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
from typing import Sequence

from utils.transform_timer import transform_timer


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class SelectChannelTransformd(transforms.MapTransform):
    """
    Transformation object that only sets every value to 0 except for
    values within channels
    """
    def __init__(self, keys: Sequence[str],
                 channels: Sequence[int] | int,
                 allow_missing_keys=False) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.channels = torch.Tensor(channels)

    def __call__(self, data: dict) -> dict:
        for key in self.keys:
            if key in data:
                mask = torch.isin(data[key], self.channels)
                data[key] = mask * data[key]
            else:
                raise ValueError(f"Key '{key}' is not in data")
        return data


class AddChanneld(transforms.MapTransform):
    def __init__(self, keys: list[str], allow_missing_keys=False) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        for key in self.keys:
            if key in data:
                data[key] = data[key].view(1, *data[key].shape)
            else:
                raise ValueError(f"Key '{key}' is not in data")
        return data


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)

    image_only = True
    if "image_only" in vars(args):
        image_only = args.image_only

    # img_keys = ["pre-image", "post-image"]
    # seg_keys = ["pre-label", "post-label"]
    img_keys = ["image"]
    seg_keys = ["label"]
    all_keys = img_keys + seg_keys

    train_transform = transforms.Compose(
        [
            transform_timer(transforms.LoadImaged(keys=all_keys)),
            transform_timer(SelectChannelTransformd(keys=seg_keys, channels=[1])),
            transform_timer(AddChanneld(keys=all_keys)),
            transform_timer(transforms.Orientationd(keys=all_keys, axcodes="RAS")),
            transform_timer(transforms.Spacingd(keys=all_keys,
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest"))),
            transform_timer(transforms.ScaleIntensityRanged(
                keys=img_keys, a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            )),
            transform_timer(transforms.RandSpatialCropSamplesd(
                keys=all_keys,
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                random_size=False,
                num_samples=4,
            )),
            transform_timer(transforms.RandFlipd(keys=all_keys, prob=args.RandFlipd_prob, spatial_axis=0)),
            transform_timer(transforms.RandFlipd(keys=all_keys, prob=args.RandFlipd_prob, spatial_axis=1)),
            transform_timer(transforms.RandFlipd(keys=all_keys, prob=args.RandFlipd_prob, spatial_axis=2)),
            transform_timer(transforms.RandRotate90d(keys=all_keys, prob=args.RandRotate90d_prob, max_k=3)),
            transform_timer(transforms.RandScaleIntensityd(keys=img_keys, factors=0.1, prob=args.RandScaleIntensityd_prob)),
            transform_timer(transforms.RandShiftIntensityd(keys=img_keys, offsets=0.1, prob=args.RandShiftIntensityd_prob)),
            transform_timer(transforms.ToTensord(keys=all_keys)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=all_keys),
            SelectChannelTransformd(keys=seg_keys, channels=[1]),
            AddChanneld(keys=all_keys),
            transforms.Orientationd(keys=all_keys, axcodes="RAS"),
            transforms.Spacingd(keys=img_keys,
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(
                keys=img_keys, a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # CROP HERE?
            transforms.ToTensord(keys=all_keys),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=all_keys, image_only=image_only),
            SelectChannelTransformd(keys=seg_keys, channels=[1]),
            AddChanneld(keys=all_keys),
            transforms.Orientationd(keys=all_keys, axcodes="RAS"),
            transforms.Spacingd(keys=img_keys,
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=img_keys, a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # CROP HERE?
            transforms.ToTensord(keys=all_keys),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
