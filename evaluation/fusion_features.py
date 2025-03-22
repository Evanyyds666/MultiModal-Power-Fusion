#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 特征融合工具

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_features(directory):
    """加载特征目录中的所有特征文件
    
    Args:
        directory: 特征文件目录
    
    Returns:
        features_dict: 包含所有特征的字典 {文件名: 特征向量}
    """
    if not os.path.exists(directory):
        raise ValueError(f"目录不存在: {directory}")
    
    features_dict = {}
    feature_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    for feature_file in feature_files:
        feature_path = os.path.join(directory, feature_file)
        try:
            feature = np.load(feature_path)
            # 如果特征是多维的，将其平坦化
            if feature.ndim > 1:
                feature = feature.flatten()
            features_dict[feature_file] = feature
        except Exception as e:
            print(f"加载特征文件 {feature_path} 失败: {e}")
    
    return features_dict

def match_features(feature_dicts):
    """匹配不同特征字典中的特征
    
    Args:
        feature_dicts: 包含多个特征字典的字典 {模态名称: {文件名: 特征向量}}
    
    Returns:
        matched_features: 包含匹配特征的字典 {文件名: {模态名称: 特征向量}}
    """
    # 提取文件名（不含扩展名）
    file_bases = {}
    for modality, features in feature_dicts.items():
        for file_name in features.keys():
            base_name = os.path.splitext(file_name)[0]
            if "_" in base_name:  # 对于有下划线的文件名，取第一个下划线之前的部分作为类别
                category = base_name.split("_")[0]
            else:
                category = "unknown"
            
            if base_name not in file_bases:
                file_bases[base_name] = {"category": category}
            
            file_bases[base_name][modality] = file_name
    
    # 匹配特征
    matched_features = {}
    
    for base_name, files in file_bases.items():
        category = files.pop("category")
        modalities_available = list(files.keys())
        
        # 只处理至少有两种模态的样本
        if len(modalities_available) >= 1:
            matched_features[base_name] = {"category": category}
            
            for modality in modalities_available:
                file_name = files[modality]
                matched_features[base_name][modality] = feature_dicts[modality][file_name]
    
    return matched_features

def fuse_features(matched_features, method="concat", output_dir=None, visualize=False):
    """融合多模态特征
    
    Args:
        matched_features: 包含匹配特征的字典 {文件名: {模态名称: 特征向量}}
        method: 融合方法，可选 'concat', 'average', 'weighted'
        output_dir: 输出目录
        visualize: 是否可视化
    
    Returns:
        fused_features: 融合后的特征字典 {文件名: 特征向量}
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    fused_features = {}
    categories = {}
    
    for base_name, features in tqdm(matched_features.items(), desc="融合特征"):
        modalities = list(features.keys())
        if "category" in modalities:
            modalities.remove("category")
            categories[base_name] = features["category"]
        
        if not modalities:
            continue
        
        # 提取特征列表
        feature_list = [features[modality] for modality in modalities]
        
        # 融合特征
        if method == "concat":
            # 如果是连接融合，需要先将所有特征标准化到相同的比例
            normalized_features = []
            for feature in feature_list:
                if np.std(feature) > 0:
                    normalized = (feature - np.mean(feature)) / np.std(feature)
                else:
                    normalized = feature
                normalized_features.append(normalized)
            
            fused = np.concatenate(normalized_features)
        
        elif method == "average":
            # 所有特征必须具有相同的维度
            if len(set(f.shape[0] for f in feature_list)) > 1:
                print(f"警告: 样本 {base_name} 的特征维度不一致，无法使用average融合")
                continue
            
            fused = np.mean(feature_list, axis=0)
        
        elif method == "weighted":
            # 所有特征必须具有相同的维度
            if len(set(f.shape[0] for f in feature_list)) > 1:
                print(f"警告: 样本 {base_name} 的特征维度不一致，无法使用weighted融合")
                continue
            
            # 使用自定义权重（可以根据先验知识调整）
            weights = {
                "clip": 0.4,
                "yolo": 0.2,
                "audio": 0.2,
                "text": 0.2
            }
            
            # 默认权重为平均权重
            default_weight = 1.0 / len(modalities)
            weighted_features = []
            
            for i, modality in enumerate(modalities):
                weight = weights.get(modality, default_weight)
                weighted_features.append(feature_list[i] * weight)
            
            fused = np.sum(weighted_features, axis=0)
        
        fused_features[base_name] = fused
        
        # 保存融合特征
        if output_dir:
            output_path = os.path.join(output_dir, f"{base_name}_fused.npy")
            np.save(output_path, fused)
    
    # 可视化
    if visualize and len(fused_features) > 3:
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
        except ImportError:
            print("警告: 无法进行可视化，需要安装scikit-learn")
            return fused_features
        
        # 收集所有融合特征
        X = np.array(list(fused_features.values()))
        
        # 收集类别
        y = [categories.get(name, "unknown") for name in fused_features.keys()]
        unique_categories = list(set(y))
        
        # PCA可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        for i, category in enumerate(unique_categories):
            mask = [label == category for label in y]
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=category)
        
        plt.title(f'PCA 可视化 ({method} 融合)')
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"pca_visualization_{method}.png"))
        
        # t-SNE可视化
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        for i, category in enumerate(unique_categories):
            mask = [label == category for label in y]
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=category)
        
        plt.title(f't-SNE 可视化 ({method} 融合)')
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"tsne_visualization_{method}.png"))
    
    return fused_features

def main():
    parser = argparse.ArgumentParser(description="融合多模态特征")
    parser.add_argument("--clip", type=str, help="CLIP特征目录")
    parser.add_argument("--yolo", type=str, help="YOLO特征目录")
    parser.add_argument("--audio", type=str, help="音频特征目录")
    parser.add_argument("--text", type=str, help="文本特征目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--method", type=str, default="concat", choices=["concat", "average", "weighted"], help="融合方法")
    parser.add_argument("--visualize", action="store_true", help="是否可视化")
    
    args = parser.parse_args()
    
    # 至少需要一种特征
    if not any([args.clip, args.yolo, args.audio, args.text]):
        print("错误: 至少需要提供一种特征目录")
        return
    
    # 加载特征
    feature_dicts = {}
    
    if args.clip:
        print(f"加载CLIP特征: {args.clip}")
        feature_dicts["clip"] = load_features(args.clip)
    
    if args.yolo:
        print(f"加载YOLO特征: {args.yolo}")
        feature_dicts["yolo"] = load_features(args.yolo)
    
    if args.audio:
        print(f"加载音频特征: {args.audio}")
        feature_dicts["audio"] = load_features(args.audio)
    
    if args.text:
        print(f"加载文本特征: {args.text}")
        feature_dicts["text"] = load_features(args.text)
    
    # 匹配特征
    print("匹配特征...")
    matched_features = match_features(feature_dicts)
    
    # 融合特征
    print(f"使用 {args.method} 方法融合特征...")
    fused_features = fuse_features(matched_features, args.method, args.output, args.visualize)
    
    print(f"融合完成，已处理 {len(fused_features)} 个样本，结果保存到 {args.output}")

if __name__ == "__main__":
    main() 