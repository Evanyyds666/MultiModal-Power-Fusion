#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 评估模型性能

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_features(feature_dir):
    """加载融合特征
    
    Args:
        feature_dir: 特征目录
    
    Returns:
        X: 特征矩阵
        y: 标签
        file_names: 文件名列表
    """
    if not os.path.exists(feature_dir):
        raise ValueError(f"特征目录不存在: {feature_dir}")
    
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    
    X = []
    y = []
    file_names = []
    
    for feature_file in feature_files:
        try:
            # 获取类别（假设第一个下划线前的部分是类别）
            base_name = os.path.splitext(feature_file)[0]
            if "_" in base_name:
                category = base_name.split("_")[0]
            else:
                # 如果文件名不包含下划线，尝试其他分割方法，或使用unknown
                category = "unknown"
            
            # 加载特征
            feature_path = os.path.join(feature_dir, feature_file)
            feature = np.load(feature_path)
            
            X.append(feature)
            y.append(category)
            file_names.append(feature_file)
        except Exception as e:
            print(f"加载 {feature_file} 时出错: {e}")
    
    return np.array(X), np.array(y), file_names

def train_and_evaluate(X, y, model_type='svm', test_size=0.2, random_state=42):
    """训练并评估模型
    
    Args:
        X: 特征矩阵
        y: 标签
        model_type: 模型类型 ('svm', 'rf', 'mlp')
        test_size: 测试集比例
        random_state: 随机数种子
    
    Returns:
        model: 训练好的模型
        y_pred: 预测结果
        y_test: 测试集标签
        X_test: 测试集特征
    """
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 创建模型
    if model_type == 'svm':
        model = SVC(probability=True)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 测试模型
    y_pred = model.predict(X_test)
    
    return model, y_pred, y_test, X_test

def print_evaluation_report(y_test, y_pred, class_names=None):
    """打印评估报告
    
    Args:
        y_test: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print("\n===== 评估报告 =====")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存混淆矩阵图
    plt.savefig('confusion_matrix.png')
    print(f"混淆矩阵已保存到 confusion_matrix.png")

def visualize_features(X, y, output_dir=None):
    """可视化特征
    
    Args:
        X: 特征矩阵
        y: 标签
        output_dir: 输出目录
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 降维可视化
    try:
        # PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(y)
        
        for label in unique_labels:
            mask = y == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label)
        
        plt.title('PCA 特征可视化')
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "pca_features.png"))
            print(f"PCA特征可视化已保存到 {os.path.join(output_dir, 'pca_features.png')}")
        
        # t-SNE降维
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        
        for label in unique_labels:
            mask = y == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label)
        
        plt.title('t-SNE 特征可视化')
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "tsne_features.png"))
            print(f"t-SNE特征可视化已保存到 {os.path.join(output_dir, 'tsne_features.png')}")
    
    except Exception as e:
        print(f"可视化特征时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="评估模型性能")
    parser.add_argument("--feature_dir", type=str, required=True, help="融合特征目录")
    parser.add_argument("--model", type=str, default="svm", choices=["svm", "rf", "mlp"], help="模型类型")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="输出目录")
    parser.add_argument("--visualize", action="store_true", help="是否可视化特征")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载特征
    print(f"从 {args.feature_dir} 加载特征...")
    X, y, file_names = load_features(args.feature_dir)
    
    # 可视化特征
    if args.visualize:
        print("可视化特征...")
        visualize_features(X, y, args.output_dir)
    
    # 训练和评估模型
    print(f"使用 {args.model} 模型进行训练和评估...")
    model, y_pred, y_test, X_test = train_and_evaluate(X, y, 
                                                       model_type=args.model, 
                                                       test_size=args.test_size)
    
    # 打印评估报告
    class_names = list(np.unique(y))
    print_evaluation_report(y_test, y_pred, class_names)
    
    # 保存结果
    results = {
        'file': [], 
        'true_label': [], 
        'predicted_label': [], 
        'correct': []
    }
    
    # 获取测试集索引
    _, X_test_indices = train_test_split(range(len(X)), test_size=args.test_size, random_state=42, stratify=y)
    
    # 生成结果
    for i, idx in enumerate(X_test_indices):
        results['file'].append(file_names[idx])
        results['true_label'].append(y_test[i])
        results['predicted_label'].append(y_pred[i])
        results['correct'].append(y_test[i] == y_pred[i])
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"评估结果已保存到 {results_csv_path}")
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for cls in class_names:
        cls_mask = results_df['true_label'] == cls
        cls_df = results_df[cls_mask]
        accuracy = cls_df['correct'].mean()
        class_accuracies[cls] = accuracy
    
    # 输出每个类别的准确率
    print("\n各类别准确率:")
    for cls, acc in class_accuracies.items():
        print(f"{cls}: {acc:.4f}")
    
    print(f"\n评估完成，结果已保存到 {args.output_dir}")

if __name__ == "__main__":
    main() 