# BiSeNet face-parsing architecture, vendored from zllrunning/face-parsing.PyTorch
# so this module has no dependency on cloning that repo on Kaggle. Layer names
# and structure are kept identical to the original so the public 79999_iter.pth
# checkpoint loads directly via load_state_dict.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

RESNET18_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"


# --------------------------------------------------------------------------
# RESNET18 BACKBONE
# --------------------------------------------------------------------------
def conv3x3(in_chan, out_chan, stride=1):
    return nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(shortcut + residual)


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    layers += [BasicBlock(out_chan, out_chan, stride=1) for _ in range(bnum - 1)]
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self._init_imagenet_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

    def _init_imagenet_weights(self):
        # Best-effort ImageNet pretraining; harmless if it fails, since
        # load_bisenet_model() below loads the full fine-tuned face-parsing
        # checkpoint over these weights anyway.
        try:
            pretrained = model_zoo.load_url(RESNET18_URL)
            own_state = self.state_dict()
            own_state.update({k: v for k, v in pretrained.items() if "fc" not in k})
            self.load_state_dict(own_state)
        except Exception as exc:
            print(f"Skipping ResNet18 ImageNet init ({exc}); full BiSeNet weights will be loaded anyway.")


# --------------------------------------------------------------------------
# BISENET BUILDING BLOCKS
# --------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return torch.mul(feat, atten)


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]

        avg = self.conv_avg(F.avg_pool2d(feat32, feat32.size()[2:]))
        avg_up = F.interpolate(avg, feat32.size()[2:], mode="nearest")

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(F.interpolate(feat32_sum, (h16, w16), mode="nearest"))

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(F.interpolate(feat16_sum, (h8, w8), mode="nearest"))

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid(self.conv2(self.relu(self.conv1(atten))))
        return feat + torch.mul(feat, atten)


# --------------------------------------------------------------------------
# BISENET (n_classes=19 for CelebAMask-HQ face parsing)
# --------------------------------------------------------------------------
class BiSeNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        h, w = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = F.interpolate(self.conv_out(feat_fuse), (h, w), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(self.conv_out16(feat_cp8), (h, w), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(self.conv_out32(feat_cp16), (h, w), mode="bilinear", align_corners=True)

        # Only feat_out (full resolution, main head) is used at inference;
        # the other two are auxiliary heads used during the original training.
        return feat_out, feat_out16, feat_out32
