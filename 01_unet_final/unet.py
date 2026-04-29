#!/usr/bin/env python3
"""
Plain U-Net baseline for BEV reconstruction.

Nothing fancy here. I just want one clean baseline first before moving to
GAN / diffusion stuff.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=8, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(ConvBlock(prev_channels, feature))
            prev_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        prev_channels = features[-1] * 2

        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2))
            self.decoder_blocks.append(ConvBlock(feature * 2, feature))
            prev_channels = feature

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        _, _, height, width = x.shape
        factor = 2 ** len(self.encoder_blocks)
        pad_h = (factor - height % factor) % factor
        pad_w = (factor - width % factor) % factor

        if pad_h > 0 or pad_w > 0:
            # 500 is not divisible by 16, so pad first and crop back later.
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        skips = []
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            skip = skips[idx]
            if x.shape[2:] != skip.shape[2:]:
                # Just in case odd shapes show up, keep decoder side aligned.
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        x = self.final(x)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :height, :width]

        return torch.sigmoid(x)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def parse_features(text: str):
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Quick U-Net sanity check.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--features", type=str, default="64,128,256,512")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    features = parse_features(args.features)
    model = UNet(in_channels=16, out_channels=8, features=features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(args.batch_size, 16, args.height, args.width, device=device)

    print(f"Device: {device}")
    print(f"Features: {features}")
    print(f"U-Net parameters: {count_parameters(model):,}")
    print(f"Input:  {x.shape}")

    with torch.no_grad():
        y = model(x)

    print(f"Output: {y.shape}")
    print(f"Range:  [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Param memory: ~{count_parameters(model) * 4 / 1024 ** 2:.0f} MB")
    if torch.cuda.is_available():
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
    print("OK!")
