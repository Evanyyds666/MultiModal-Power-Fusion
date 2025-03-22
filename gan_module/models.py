#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GAN模型定义

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """U-Net生成器网络"""
    
    def __init__(self, input_channels=3, output_channels=3, n_down=6):
        """
        参数:
            input_channels: 输入通道数
            output_channels: 输出通道数
            n_down: 下采样层数
        """
        super(UNetGenerator, self).__init__()
        
        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 下采样部分
        mult = 1
        self.down_layers = nn.ModuleList()
        for i in range(n_down - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), 8)
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(64 * mult_prev, 64 * mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64 * mult),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        # 上采样部分
        self.up_layers = nn.ModuleList()
        for i in range(n_down - 1):
            mult_prev = mult
            mult = min(2 ** (n_down - i - 2), 8) if i < n_down - 1 else 1
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(64 * mult_prev * 2, 64 * mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64 * mult),
                nn.ReLU(inplace=True)
            ))
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 下采样和存储中间结果
        results = [self.init_conv(x)]
        for down in self.down_layers:
            results.append(down(results[-1]))
        
        # 上采样并连接跳跃连接
        out = results[-1]
        for i, up in enumerate(self.up_layers):
            out = up(torch.cat([out, results[-(i+2)]], dim=1))
        
        return self.final(torch.cat([out, results[0]], dim=1))

class PatchDiscriminator(nn.Module):
    """PatchGAN判别器网络"""
    
    def __init__(self, input_channels=6):
        """
        参数:
            input_channels: 输入通道数 (正常图像+故障图像)
        """
        super(PatchDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 第一层不使用BN
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, normal, fault):
        # 连接输入图像和目标图像/生成图像
        x = torch.cat([normal, fault], dim=1)
        return self.model(x) 