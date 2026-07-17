"""Định nghĩa model Stage 1 (EfficientNet corner regression) — port từ full_pipeline.ipynb."""

import timm
import torch
import torch.nn as nn


class EfficientNetCorner(nn.Module):
    """EfficientNet-B3 backbone + regression head xuất 4 corner (8 giá trị, sigmoid).

    Input: ảnh grayscale 1 kênh. gray_to_rgb (conv 1x1) nhân bản thành 3 kênh
    trước khi đưa vào backbone timm.
    """

    def __init__(self, variant='b3', dropout=0.0):
        super().__init__()
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.gray_to_rgb.weight, 1.0 / 3.0)
        self.backbone = timm.create_model(
            f'efficientnet_{variant}', pretrained=False,
            num_classes=0, global_pool='avg',
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.SiLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(512, 256), nn.SiLU(inplace=True), nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, 8), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.backbone(self.gray_to_rgb(x)))


def order_corners_tl_tr_br_bl(pts):
    """Sort 4 (x,y) points → TL, TR, BR, BL (consistent warp target)."""
    by_y = sorted(pts, key=lambda p: p[1])
    top = sorted(by_y[:2], key=lambda p: p[0])
    bot = sorted(by_y[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[1], bot[0]]
