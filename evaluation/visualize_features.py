#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from pathlib import Path
import glob
from tqdm import tqdm

def load_features(features_dir):
    """加载特征文件
    
    Args:
        features_dir: 特征文件目录
        
    Returns:
        特征数组和对应的文件名
    """
    feature_files = glob.glob(os.path.join(features_dir, "*.npy"))
    features = []
    filenames = []
    
    for file_path in tqdm(feature_files, desc="加载特征"):
        try:
            feature = np.load(file_path)
            features.append(feature)
            filenames.append(os.path.basename(file_path))
        except Exception as e:
            print(f"加载特征失败 {file_path}: {e}")
    
    return np.array(features), filenames

def reduce_dimensions(features, method='tsne', n_components=2):
    """降维处理
    
    Args:
        features: 特征数组
        method: 降维方法 ('tsne' 或 'pca')
        n_components: 降维后的维度
        
    Returns:
        降维后的特征
    """
    if len(features) < 2:
        raise ValueError("至少需要两个特征样本进行降维")
    
    # 处理特征形状
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    
    if method.lower() == 'tsne':
        # 如果特征维度很高并且样本数量足够，先用PCA降维以加速t-SNE
        if features.shape[1] > 50 and len(features) > 50:
            pca_n_components = min(50, len(features) - 1)
            pca = PCA(n_components=pca_n_components)
            features = pca.fit_transform(features)
            print(f"使用PCA将特征降至{pca_n_components}维，以加速t-SNE")
        
        # 设置perplexity小于样本数量
        perplexity = min(30, len(features) - 1) if len(features) > 3 else 1
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        return tsne.fit_transform(features)
    
    elif method.lower() == 'pca':
        pca = PCA(n_components=min(n_components, len(features) - 1))
        return pca.fit_transform(features)
    
    else:
        raise ValueError(f"不支持的降维方法: {method}")

def visualize_features(features_dir, output_path, method='tsne', n_components=2):
    """可视化特征
    
    Args:
        features_dir: 特征文件目录
        output_path: 输出图像路径
        method: 降维方法 ('tsne' 或 'pca')
        n_components: 降维后的维度
    """
    # 加载特征
    features, filenames = load_features(features_dir)
    
    if len(features) == 0:
        print(f"警告: 目录 {features_dir} 中未找到特征文件")
        return
    
    print(f"加载了 {len(features)} 个特征向量")
    
    # 降维处理
    reduced_features = reduce_dimensions(features, method, n_components)
    
    # 创建可视化
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
        
        # 如果特征数量不多，添加文件名标签
        if len(features) <= 20:
            for i, filename in enumerate(filenames):
                plt.annotate(os.path.splitext(filename)[0], 
                            (reduced_features[i, 0], reduced_features[i, 1]),
                            fontsize=9)
    
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], alpha=0.7)
        
        # 如果特征数量不多，添加文件名标签
        if len(features) <= 20:
            for i, filename in enumerate(filenames):
                ax.text(reduced_features[i, 0], reduced_features[i, 1], reduced_features[i, 2],
                        os.path.splitext(filename)[0], fontsize=9)
    
    plt.title(f"使用 {method.upper()} 可视化特征")
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"可视化结果已保存到 {output_path}")

def main():
    parser = argparse.ArgumentParser(description="可视化特征向量")
    parser.add_argument("--features", type=str, required=True,
                        help="包含特征文件的目录")
    parser.add_argument("--output", type=str, default="feature_visualization.png",
                        help="输出图像路径")
    parser.add_argument("--method", type=str, default="tsne",
                        choices=["tsne", "pca"],
                        help="降维方法")
    parser.add_argument("--components", type=int, default=2,
                        choices=[2, 3],
                        help="降维后的维度")
    
    args = parser.parse_args()
    
    visualize_features(
        args.features, 
        args.output, 
        args.method, 
        args.components
    )

if __name__ == "__main__":
    main() 