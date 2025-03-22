#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 音频特征提取模块

import os
import argparse
import numpy as np
import librosa
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

def extract_audio_features(model, feature_extractor, audio_path, output_dir, sampling_rate=16000, max_duration=10):
    """提取音频特征
    
    Args:
        model: 音频模型
        feature_extractor: 特征提取器
        audio_path: 音频文件或目录路径
        output_dir: 输出特征的目录
        sampling_rate: 采样率
        max_duration: 最大音频处理时长（秒）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理输入路径
    if os.path.isfile(audio_path):
        audio_files = [audio_path]
    else:
        audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) 
                      if f.lower().endswith(('wav', 'mp3', 'ogg', 'flac'))]
    
    for audio_file in tqdm(audio_files, desc="提取音频特征"):
        try:
            # 加载音频
            audio, sr = librosa.load(audio_file, sr=sampling_rate, mono=True)
            
            # 限制音频长度
            if len(audio) > max_duration * sampling_rate:
                audio = audio[:max_duration * sampling_rate]
            
            # 提取频谱特征
            inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 提取embeddings特征
            with torch.no_grad():
                outputs = model(**inputs)
                # 获取最后一层隐藏状态作为特征
                if hasattr(outputs, "hidden_states"):
                    # 使用最后一层隐藏状态的平均值
                    audio_features = outputs.hidden_states[-1].mean(dim=1)
                else:
                    # 如果没有隐藏状态，使用logits
                    audio_features = outputs.logits
                
                # 归一化特征
                audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
            
            # 保存特征
            audio_name = os.path.basename(audio_file)
            feature_name = os.path.splitext(audio_name)[0] + ".npy"
            feature_path = os.path.join(output_dir, feature_name)
            
            # 保存为numpy格式
            np.save(feature_path, audio_features[0].cpu().numpy())
            
        except Exception as e:
            print(f"处理音频失败 {audio_file}: {e}")
    
    print(f"音频特征提取完成，已保存到 {output_dir}")

def extract_traditional_features(audio_path, output_dir):
    """使用传统方法提取音频特征(MFCC, 色度, 梅尔频谱等)
    
    Args:
        audio_path: 音频文件或目录路径
        output_dir: 输出特征的目录
    """
    try:
        import librosa
        import librosa.display
    except ImportError:
        print("错误: 需要安装librosa库才能提取传统音频特征")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理输入路径
    if os.path.isfile(audio_path):
        audio_files = [audio_path]
    else:
        audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) 
                      if f.lower().endswith(('wav', 'mp3', 'ogg', 'flac'))]
    
    if not audio_files:
        print(f"警告: 在 {audio_path} 中没有找到音频文件")
        return
    
    # 处理每个音频文件
    for audio_file in tqdm(audio_files, desc="提取传统音频特征"):
        try:
            # 加载音频文件
            y, sr = librosa.load(audio_file, sr=None)
            
            # 提取特征
            # 1. MFCC特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 2. 色度特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 3. 梅尔频谱
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # 4. 光谱对比度
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # 5. 光谱质心
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            
            # 将特征压缩为固定长度（取平均值）
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
            chroma_mean = np.mean(chroma, axis=1)
            log_mel_mean = np.mean(log_mel_spec, axis=1)
            contrast_mean = np.mean(contrast, axis=1)
            centroid_mean = np.mean(centroid, axis=1)
            
            # 合并特征
            feature_vector = np.concatenate([
                mfcc_mean, mfcc_delta_mean, mfcc_delta2_mean,
                chroma_mean, log_mel_mean, contrast_mean, centroid_mean
            ])
            
            # 保存特征
            audio_name = os.path.basename(audio_file)
            feature_name = os.path.splitext(audio_name)[0] + ".npy"
            feature_path = os.path.join(output_dir, feature_name)
            
            np.save(feature_path, feature_vector)
            
        except Exception as e:
            print(f"处理音频 {audio_file} 时出错: {e}")
    
    print(f"传统音频特征提取完成，已保存到 {output_dir}")

def extract_wavvec_features(audio_path, output_dir, model_name='facebook/wav2vec2-base-960h'):
    """使用预训练深度学习模型提取音频特征
    
    Args:
        audio_path: 音频文件或目录路径
        output_dir: 输出特征的目录
        model_name: 使用的预训练模型名称
    """
    try:
        import librosa
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
    except ImportError:
        print("错误: 需要安装transformers和librosa库才能提取wav2vec特征")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预训练模型和处理器
    print(f"加载模型: {model_name}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    
    # 移动模型到适当的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # 处理输入路径
    if os.path.isfile(audio_path):
        audio_files = [audio_path]
    else:
        audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) 
                      if f.lower().endswith(('wav', 'mp3', 'ogg', 'flac'))]
    
    if not audio_files:
        print(f"警告: 在 {audio_path} 中没有找到音频文件")
        return
    
    # 处理每个音频文件
    for audio_file in tqdm(audio_files, desc="提取wav2vec音频特征"):
        try:
            # 加载音频文件
            speech_array, sampling_rate = librosa.load(audio_file, sr=16000)
            
            # 预处理音频
            inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用最后一层隐藏状态
                last_hidden_states = outputs.last_hidden_state
                
                # 取平均值得到固定长度的特征向量
                feature_vector = last_hidden_states.mean(dim=1).cpu().numpy()[0]
            
            # 保存特征
            audio_name = os.path.basename(audio_file)
            feature_name = os.path.splitext(audio_name)[0] + ".npy"
            feature_path = os.path.join(output_dir, feature_name)
            
            np.save(feature_path, feature_vector)
            
        except Exception as e:
            print(f"处理音频 {audio_file} 时出错: {e}")
    
    print(f"wav2vec音频特征提取完成，已保存到 {output_dir}")

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="提取音频特征")
    parser.add_argument("--input", type=str, required=True, 
                        help="音频文件或目录路径")
    parser.add_argument("--output", type=str, default="audio_features", 
                        help="输出特征目录")
    parser.add_argument("--method", type=str, choices=["traditional", "wav2vec"], default="wav2vec",
                        help="特征提取方法")
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-base-960h",
                        help="wav2vec模型名称或路径")
    
    args = parser.parse_args()
    
    if args.method == "traditional":
        print("使用传统方法提取音频特征...")
        extract_traditional_features(args.input, args.output)
    else:
        print(f"使用 {args.model} 提取wav2vec特征...")
        extract_wavvec_features(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
