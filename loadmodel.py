import torch

from monai.networks.nets import SwinUNETR


def main():
    torch.random.manual_seed(0)

    out_channels = 2
    feature_size = 48

    model = SwinUNETR(
        img_size=96,
        in_channels=1,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    )

    model_dict = torch.load("pretrained_models/swin_unetr_1_channel.pt",
                            map_location="cpu")

    out_kernel_key = "out.conv.conv.weight"
    out_biases_key = "out.conv.conv.bias"

    model_dict["state_dict"][out_kernel_key] = torch.randn(
        (out_channels, feature_size, 1, 1, 1)
    )
    model_dict["state_dict"][out_biases_key] = torch.randn(out_channels)

    torch.save(model_dict, f"pretrained_models/swin_unetr_{out_channels}_channel.pt")


if __name__ == '__main__':
    main()
