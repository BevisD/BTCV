import csv
import os
import argparse

import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

from utils.data_utils import get_loader
from utils.utils import resample_3d

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.jsons", type=str, help="dataset jsons file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-150.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=200.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.0, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.0, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.0, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.0, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--save_outputs", action="store_true", help="save output segmentations")


def main():
    args = parser.parse_args()
    args.test_mode = True
    args.image_only = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    val_loader = get_loader(args)

    dice_acc = DiceMetric(include_background=False, get_not_nans=True)

    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    model = SwinUNETR(
        img_size=96,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    metrics = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = batch["image"], batch["label"]
            val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)

            original_affine = batch["label_meta_dict"]["affine"][0].numpy()

            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            pre_post = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-3]
            print(f"Inference on case {pre_post}/{img_name}")
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )

            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)

            val_outputs = val_outputs.cpu().numpy().astype(np.uint8)[0, 0]
            val_outputs = resample_3d(val_outputs, target_shape)
            val_outputs = torch.Tensor(val_outputs).view((1, 1, *val_outputs.shape)).cuda()

            dice_acc.reset()
            dice_acc(y_pred=val_outputs, y=val_labels)
            acc, not_nans = dice_acc.aggregate()
            print(f"Mean Site Dice: {acc.item():.4f} Not Nans # {not_nans.item():.0f}")

            if args.save_outputs:
                nifti_image = nib.Nifti1Image(
                    val_outputs.cpu().numpy().astype(np.int8)[0, 0],
                    original_affine
                )
                nib.save(
                    nifti_image,
                    os.path.join(output_directory, img_name)
                )

            metrics.append({
                "pre_post": pre_post,
                "filename": img_name,
                "dice": acc.item(),
                "nan": not bool(not_nans.item())
            })

    with open(os.path.join(output_directory, "metrics.csv"), "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metrics[0].keys()))
        writer.writeheader()

        for metric in metrics:
            writer.writerow(metric)


if __name__ == '__main__':
    main()
