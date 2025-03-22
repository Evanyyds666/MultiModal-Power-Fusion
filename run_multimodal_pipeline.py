#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 多模态特征提取和融合流水线

import os
import argparse
import numpy as np
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_module(module_name):
    """检查模块是否可用"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def extract_image_features(image_path, output_dir, clip_model='ViT-B-16', yolo_model=None):
    """提取图像特征
    
    使用CN-CLIP提取语义特征和YOLO提取目标检测特征
    
    Args:
        image_path: 图像或图像目录路径
        output_dir: 输出目录
        clip_model: CLIP模型名称
        yolo_model: YOLO模型路径
    """
    clip_output_dir = os.path.join(output_dir, "clip_features")
    os.makedirs(clip_output_dir, exist_ok=True)
    
    # 使用CN-CLIP提取特征
    if check_module("cn_clip"):
        print(f"使用CN-CLIP ({clip_model}) 提取图像特征...")
        from Feature_extraction.clip_feature_extraction import extract_features
        from cn_clip.clip import load_from_name
        
        model, preprocess = load_from_name(clip_model)
        extract_features(model, preprocess, image_path, clip_output_dir)
    else:
        print("警告: cn_clip 模块不可用，跳过CLIP特征提取")
    
    # 使用YOLO提取特征
    if yolo_model and check_module("ultralytics"):
        yolo_output_dir = os.path.join(output_dir, "yolo_features")
        os.makedirs(yolo_output_dir, exist_ok=True)
        
        print(f"使用YOLO模型提取目标检测特征...")
        from ultralytics import YOLO
        from Feature_extraction.yolo_feature_extraction import extract_yolo_features
        
        model = YOLO(yolo_model)
        extract_yolo_features(model, image_path, yolo_output_dir)
    
    return clip_output_dir

def extract_audio_features(audio_path, output_dir, model_name='facebook/wav2vec2-base-960h', traditional=False):
    """提取音频特征
    
    Args:
        audio_path: 音频或音频目录路径
        output_dir: 输出目录
        model_name: 音频模型名称
        traditional: 是否使用传统方法提取特征
    """
    audio_output_dir = os.path.join(output_dir, "audio_features")
    os.makedirs(audio_output_dir, exist_ok=True)
    
    if traditional:
        if check_module("librosa"):
            print("使用传统方法提取音频特征 (MFCC, 色度, 梅尔频谱)...")
            from Feature_extraction.audio_feature_extraction import extract_traditional_features
            extract_traditional_features(audio_path, audio_output_dir)
        else:
            print("警告: librosa 模块不可用，跳过传统音频特征提取")
    else:
        if check_module("transformers") and check_module("torch"):
            print(f"使用深度学习模型 ({model_name}) 提取音频特征...")
            from Feature_extraction.audio_feature_extraction import extract_wavvec_features
            extract_wavvec_features(audio_path, audio_output_dir, model_name)
        else:
            print("警告: transformers 或 torch 模块不可用，跳过深度音频特征提取")
    
    return audio_output_dir

def extract_text_features(text_path, output_dir, model_name='bert-base-chinese'):
    """提取文本特征
    
    Args:
        text_path: 文本或文本目录路径
        output_dir: 输出目录
        model_name: 文本模型名称
    """
    text_output_dir = os.path.join(output_dir, "text_features")
    os.makedirs(text_output_dir, exist_ok=True)
    
    if check_module("transformers") and check_module("torch"):
        print(f"使用 {model_name} 提取文本特征...")
        from Feature_extraction.text_feature_extraction import extract_text_features
        extract_text_features(text_path, text_output_dir, model_name)
    else:
        print("警告: transformers 或 torch 模块不可用，跳过文本特征提取")
    
    return text_output_dir

def fuse_features(feature_dirs, output_dir, fusion_method='concat', visualize=False):
    """融合多模态特征
    
    Args:
        feature_dirs: 包含多个特征目录的字典
        output_dir: 输出目录
        fusion_method: 融合方法，可选 'concat', 'average', 'weighted'
        visualize: 是否可视化
    """
    # 检查特征目录
    valid_dirs = {k: v for k, v in feature_dirs.items() if v is not None and os.path.exists(v)}
    
    if not valid_dirs:
        print("错误: 没有有效的特征目录")
        return
    
    print(f"融合多模态特征，使用方法: {fusion_method}")
    fused_output_dir = os.path.join(output_dir, "fused_features")
    os.makedirs(fused_output_dir, exist_ok=True)
    
    # 收集所有特征文件
    features_by_sample = {}
    for modality, directory in valid_dirs.items():
        feature_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        
        for feature_file in feature_files:
            sample_id = os.path.splitext(feature_file)[0]
            feature_path = os.path.join(directory, feature_file)
            
            if sample_id not in features_by_sample:
                features_by_sample[sample_id] = {}
            
            features_by_sample[sample_id][modality] = feature_path
    
    print(f"找到 {len(features_by_sample)} 个样本进行融合")
    
    # 为每个样本融合特征
    for sample_id, modality_features in tqdm(features_by_sample.items(), desc="融合特征"):
        # 加载特征
        loaded_features = {}
        for modality, feature_path in modality_features.items():
            try:
                feature = np.load(feature_path)
                # 如果特征是多维的，将其平坦化
                if feature.ndim > 1:
                    feature = feature.flatten()
                loaded_features[modality] = feature
            except Exception as e:
                print(f"加载特征 {feature_path} 失败: {e}")
                continue
        
        if not loaded_features:
            continue
        
        # 融合特征
        if fusion_method == 'concat':
            # 如果是连接融合，需要先将所有特征标准化到相同的比例
            normalized_features = []
            for feature in loaded_features.values():
                if np.std(feature) > 0:
                    normalized = (feature - np.mean(feature)) / np.std(feature)
                else:
                    normalized = feature
                normalized_features.append(normalized)
            
            fused = np.concatenate(normalized_features)
            
        elif fusion_method == 'average':
            # 所有特征必须具有相同的维度
            if len(set(f.shape[0] for f in loaded_features.values())) > 1:
                print(f"警告: 样本 {sample_id} 的特征维度不一致，无法使用average融合")
                continue
            
            fused = np.mean(list(loaded_features.values()), axis=0)
            
        elif fusion_method == 'weighted':
            # 所有特征必须具有相同的维度
            if len(set(f.shape[0] for f in loaded_features.values())) > 1:
                print(f"警告: 样本 {sample_id} 的特征维度不一致，无法使用weighted融合")
                continue
            
            # 使用自定义权重（可以根据先验知识调整）
            weights = {
                'clip': 0.4,
                'yolo': 0.2,
                'audio': 0.2,
                'text': 0.2
            }
            
            # 默认权重为平均权重
            default_weight = 1.0 / len(loaded_features)
            weighted_features = []
            
            for modality, feature in loaded_features.items():
                weight = weights.get(modality, default_weight)
                weighted_features.append(feature * weight)
            
            fused = np.sum(weighted_features, axis=0)
        
        # 保存融合后的特征
        output_path = os.path.join(fused_output_dir, f"{sample_id}_fused.npy")
        np.save(output_path, fused)
    
    # 可视化
    if visualize and check_module("sklearn"):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        print("正在可视化融合特征...")
        
        # 收集所有融合特征
        fused_files = [os.path.join(fused_output_dir, f) for f in os.listdir(fused_output_dir)]
        if len(fused_files) > 3:  # 只有当有足够的样本时才可视化
            features = []
            labels = []
            
            for fused_file in fused_files:
                feature = np.load(fused_file)
                category = os.path.basename(fused_file).split('_')[0]  # 假设文件名格式为 'category_id_fused.npy'
                
                features.append(feature)
                labels.append(category)
            
            features = np.array(features)
            
            # PCA可视化
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(features)
            
            plt.figure(figsize=(10, 8))
            categories = set(labels)
            for category in categories:
                indices = [i for i, label in enumerate(labels) if label == category]
                plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=category, alpha=0.7)
            
            plt.title('PCA Visualization of Fused Features')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
            
            # t-SNE可视化
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(features)
            
            plt.figure(figsize=(10, 8))
            for category in categories:
                indices = [i for i, label in enumerate(labels) if label == category]
                plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=category, alpha=0.7)
            
            plt.title('t-SNE Visualization of Fused Features')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
            
            print(f"可视化结果已保存到 {output_dir}")
    
    return fused_output_dir

def main():
    parser = argparse.ArgumentParser(description="多模态特征提取和融合流水线")
    
    # 输入参数
    parser.add_argument("--images", type=str, default=None, help="图像或图像目录路径")
    parser.add_argument("--audio", type=str, default=None, help="音频或音频目录路径")
    parser.add_argument("--texts", type=str, default=None, help="文本或文本目录路径")
    parser.add_argument("--output", type=str, default="multimodal_results", help="输出目录")
    
    # 模型参数
    parser.add_argument("--clip-model", type=str, default="ViT-B-16", help="CLIP模型名称")
    parser.add_argument("--yolo-model", type=str, default=None, help="YOLO模型路径")
    parser.add_argument("--audio-model", type=str, default="facebook/wav2vec2-base-960h", help="音频模型名称")
    parser.add_argument("--text-model", type=str, default="bert-base-chinese", help="文本模型名称")
    
    # 融合参数
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "average", "weighted"], help="特征融合方法")
    parser.add_argument("--visualize", action="store_true", help="是否可视化融合特征")
    
    # 其他参数
    parser.add_argument("--traditional-audio", action="store_true", help="是否使用传统方法提取音频特征")
    parser.add_argument("--use-gan", action="store_true", help="是否使用GAN生成故障图像")
    parser.add_argument("--gan-model", type=str, default=None, help="GAN模型路径")
    
    args = parser.parse_args()
    
    # 检查输入
    if not any([args.images, args.audio, args.texts]):
        print("错误: 至少需要提供一种模态的输入 (图像, 音频或文本)")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 提取特征
    feature_dirs = {}
    
    if args.images:
        print("\n===== 提取图像特征 =====")
        if args.use_gan and args.gan_model and check_module("gan_module"):
            # 使用GAN生成故障图像
            from gan_module import generate_fault_images
            
            print(f"使用GAN生成故障图像...")
            normal_dir = args.images
            fault_dir = os.path.join(args.output, "generated_faults")
            os.makedirs(fault_dir, exist_ok=True)
            
            generate_fault_images(
                args.gan_model,
                normal_dir,
                fault_dir,
                num_variations=3,
                severity=0.7
            )
            
            # 提取正常和故障图像的特征
            clip_features_normal = extract_image_features(normal_dir, args.output, args.clip_model, args.yolo_model)
            clip_features_fault = extract_image_features(fault_dir, args.output, args.clip_model, args.yolo_model)
            
            # 合并特征目录
            feature_dirs['clip_normal'] = clip_features_normal
            feature_dirs['clip_fault'] = clip_features_fault
        else:
            # 只提取原始图像特征
            clip_features = extract_image_features(args.images, args.output, args.clip_model, args.yolo_model)
            feature_dirs['clip'] = clip_features
    
    if args.audio:
        print("\n===== 提取音频特征 =====")
        audio_features = extract_audio_features(args.audio, args.output, args.audio_model, args.traditional_audio)
        feature_dirs['audio'] = audio_features
    
    if args.texts:
        print("\n===== 提取文本特征 =====")
        text_features = extract_text_features(args.texts, args.output, args.text_model)
        feature_dirs['text'] = text_features
    
    # 融合特征
    if len(feature_dirs) > 0:
        print("\n===== 融合多模态特征 =====")
        fused_dir = fuse_features(feature_dirs, args.output, args.fusion, args.visualize)
        print(f"\n特征提取和融合完成! 结果保存在: {args.output}")
    else:
        print("错误: 没有成功提取任何特征")

if __name__ == "__main__":
    main() 