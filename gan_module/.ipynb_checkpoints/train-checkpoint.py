#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GAN训练模块

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from .models import UNetGenerator, PatchDiscriminator
from .data_utils import PowerEquipmentDataset, set_seed

def train_gan(normal_dir, fault_dir, output_dir, epochs=100, batch_size=4, lr=0.0002, 
              beta1=0.5, beta2=0.999, lambda_l1=100, image_size=256, save_interval=10):
    """
    训练图像转换GAN
    
    参数:
        normal_dir: 正常设备图像目录
        fault_dir: 故障设备图像目录（可选）
        output_dir: 输出目录（模型和样本）
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        beta1, beta2: Adam优化器参数
        lambda_l1: L1损失权重
        image_size: 图像大小
        save_interval: 保存模型和示例的间隔轮数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 数据集和数据加载器
    dataset = PowerEquipmentDataset(normal_dir, fault_dir, transform, paired=(fault_dir is not None))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 初始化模型
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    
    # 损失函数
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    # 优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # 跟踪训练过程
    losses_g = []
    losses_d = []
    
    # 训练循环
    print(f"开始训练，共 {epochs} 轮...")
    for epoch in range(epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        
        pbar = tqdm(dataloader, desc=f"训练轮次 {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            normal_images = batch['normal'].to(device)
            fault_images = batch['fault'].to(device)
            
            batch_size = normal_images.size(0)
            
            # 真实和虚假标签
            real_label = torch.ones((batch_size, 1, 30, 30), device=device)
            fake_label = torch.zeros((batch_size, 1, 30, 30), device=device)
            
            # -----------------------------
            # 训练判别器
            # -----------------------------
            optimizer_d.zero_grad()
            
            # 真实对判别
            real_output = discriminator(normal_images, fault_images)
            loss_d_real = criterion_gan(real_output, real_label)
            
            # 生成虚假故障图像
            fake_fault = generator(normal_images)
            
            # 虚假对判别
            fake_output = discriminator(normal_images, fake_fault.detach())
            loss_d_fake = criterion_gan(fake_output, fake_label)
            
            # 总判别器损失
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            # -----------------------------
            # 训练生成器
            # -----------------------------
            optimizer_g.zero_grad()
            
            # 生成器GAN损失
            fake_output = discriminator(normal_images, fake_fault)
            loss_g_gan = criterion_gan(fake_output, real_label)
            
            # 生成器L1损失
            loss_g_l1 = criterion_l1(fake_fault, fault_images) * lambda_l1
            
            # 总生成器损失
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            optimizer_g.step()
            
            # 更新进度条和累加损失
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            
            pbar.set_postfix(D_loss=f"{loss_d.item():.4f}", G_loss=f"{loss_g.item():.4f}")
        
        # 计算轮次平均损失
        avg_loss_d = epoch_loss_d / len(dataloader)
        avg_loss_g = epoch_loss_g / len(dataloader)
        losses_d.append(avg_loss_d)
        losses_g.append(avg_loss_g)
        
        print(f"轮次 {epoch+1}: 判别器损失 = {avg_loss_d:.4f}, 生成器损失 = {avg_loss_g:.4f}")
        
        # 每save_interval轮保存模型和生成样本
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
            print(f"已保存检查点: {checkpoint_path}")
            
            # 生成并保存样本
            with torch.no_grad():
                num_samples = min(4, len(normal_images))
                fake_faults = generator(normal_images[:num_samples])
                # 将tensor转换回图像格式
                for j in range(num_samples):
                    # 转回 [0, 1] 范围
                    normal = normal_images[j].cpu().detach()
                    fake = fake_faults[j].cpu().detach()
                    real_fault = fault_images[j].cpu().detach()
                    
                    normal = (normal * 0.5) + 0.5
                    fake = (fake * 0.5) + 0.5
                    real_fault = (real_fault * 0.5) + 0.5
                    
                    # 创建比较图
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(normal.permute(1, 2, 0))
                    axs[0].set_title('正常设备')
                    axs[0].axis('off')
                    
                    axs[1].imshow(fake.permute(1, 2, 0))
                    axs[1].set_title('生成的故障')
                    axs[1].axis('off')
                    
                    axs[2].imshow(real_fault.permute(1, 2, 0))
                    axs[2].set_title('实际故障(如有)')
                    axs[2].axis('off')
                    
                    sample_path = os.path.join(samples_dir, f"epoch_{epoch+1}_sample_{j}.png")
                    plt.savefig(sample_path)
                    plt.close()
                    print(f"已保存样本: {sample_path}")
    
    # 保存最终模型
    final_generator_path = os.path.join(output_dir, "generator_final.pth")
    final_discriminator_path = os.path.join(output_dir, "discriminator_final.pth")
    torch.save(generator.state_dict(), final_generator_path)
    torch.save(discriminator.state_dict(), final_discriminator_path)
    print(f"训练完成，最终模型已保存到: {final_generator_path}, {final_discriminator_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), losses_g, label='生成器损失')
    plt.plot(range(1, epochs+1), losses_d, label='判别器损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练损失')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    return generator, discriminator 