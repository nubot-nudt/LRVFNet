import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------#
#   Mish activation (paper: DarknetConv2D_BN_Mish)
# -------------------------------------------------#
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BasicConv(nn.Module):
    """Conv2D + BN + Mish, the building block of CSPDarknet53."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3),
        )

    def forward(self, x):
        return x + self.block(x)


class Resblock_body(nn.Module):
    """CSP residual stage: one downsample + a CSP split/concat around n Resblocks."""

    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels // 2),
                BasicConv(out_channels, out_channels, 1),
            )
            self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels // 2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels // 2) for _ in range(num_blocks)],
                BasicConv(out_channels // 2, out_channels // 2, 1),
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.blocks_conv(self.split_conv1(x))
        return self.concat_conv(torch.cat([x1, x0], dim=1))


class CSPDarkNet(nn.Module):
    """CSPDarknet53 as described in the paper (Resblock_body x 2/8/8/4).

    Returns the three feature maps consumed by the fusion module:
        80x80x512, 40x40x1024, 20x20x1024   (input 640x640)
    """

    def __init__(self, layers=(1, 2, 8, 8, 4)):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False),
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)
        return out3, out4, out5


# Public CSPDarknet53 backbone weights.
#
# Original CSPDarknet53 (CSPNet, CVPR 2020) ImageNet weights are released as
# Darknet `.weights` in https://github.com/WongKinYiu/CrossStagePartialNetworks.
# The PyTorch-converted checkpoint used here comes from the MIT-licensed
# https://github.com/bubbliiiing/yolov4-pytorch (release v1.0); we mirror the file
# on the Hugging Face Hub so the link stays stable. Keys are `conv1.*` + `stages.*`
# and match CSPDarkNet below one-to-one.
CSPDARKNET53_BACKBONE_URL = (
    "https://huggingface.co/yuhan-xiao/LRVFNet/resolve/main/"
    "CSPdarknet53_backbone_weights.pth"
)
# Mirror used if the primary URL is unreachable.
CSPDARKNET53_BACKBONE_URL_FALLBACK = (
    "https://github.com/bubbliiiing/yolov4-pytorch/releases/download/v1.0/"
    "CSPdarknet53_backbone_weights.pth"
)


def darknetradar(pretrained=False, **kwargs):
    """Build CSPDarknet53 for the BEV branch.

    pretrained=True/'default' -> download the public backbone weights above.
    pretrained='<path>'       -> load that local checkpoint instead.
    """
    model = CSPDarkNet()
    if not pretrained:
        return model

    if isinstance(pretrained, str) and pretrained not in ("default", "url"):
        state = torch.load(pretrained, map_location="cpu")
        src = pretrained
    else:
        try:
            state = torch.hub.load_state_dict_from_url(
                CSPDARKNET53_BACKBONE_URL, map_location="cpu", model_dir="./model_data")
            src = "HuggingFace:" + CSPDARKNET53_BACKBONE_URL.split("/")[-1]
        except Exception as e:
            print("HF download failed (%s), falling back to the original release." % e)
            state = torch.hub.load_state_dict_from_url(
                CSPDARKNET53_BACKBONE_URL_FALLBACK, map_location="cpu", model_dir="./model_data")
            src = "GitHub:" + CSPDARKNET53_BACKBONE_URL_FALLBACK.split("/")[-1]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("CSPDarknet53 backbone: loaded %s (missing=%d, unexpected=%d)"
          % (src, len(missing), len(unexpected)))
    return model
