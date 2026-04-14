#!/usr/bin/env python3
"""
3D Gaussian Splatting 模型后处理优化脚本（修复版）
处理Sigmoid激活前的原始值
"""

import numpy as np
from plyfile import PlyData, PlyElement
import argparse
from pathlib import Path

def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def analyze_model(ply_path):
    """分析模型文件"""
    print(f"分析模型: {ply_path}")
    
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex'].data
    
    print(f"高斯点总数: {len(vertices):,}")
    
    # 不透明度分析（原始值，需要Sigmoid）
    if 'opacity' in vertices.dtype.names:
        opacities_raw = vertices['opacity']
        opacities_sigmoid = sigmoid(opacities_raw)
        
        print(f"\n不透明度统计 (原始值):")
        print(f"  平均值: {np.mean(opacities_raw):.6f}")
        print(f"  中位数: {np.median(opacities_raw):.6f}")
        print(f"  最小值: {np.min(opacities_raw):.6f}")
        print(f"  最大值: {np.max(opacities_raw):.6f}")
        
        print(f"\n不透明度统计 (Sigmoid后):")
        print(f"  平均值: {np.mean(opacities_sigmoid):.6f}")
        print(f"  中位数: {np.median(opacities_sigmoid):.6f}")
        print(f"  最小值: {np.min(opacities_sigmoid):.6f}")
        print(f"  最大值: {np.max(opacities_sigmoid):.6f}")
        
        # 分布统计 (Sigmoid后)
        bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
        for low, high in bins:
            count = np.sum((opacities_sigmoid >= low) & (opacities_sigmoid < high))
            percentage = count / len(opacities_sigmoid) * 100
            print(f"  [{low:.2f}-{high:.2f}): {count:,} ({percentage:.1f}%)")
    
    # 缩放分析（原始值，需要exp）
    if all(k in vertices.dtype.names for k in ['scale_0', 'scale_1', 'scale_2']):
        scales_raw = np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T
        scales_exp = np.exp(scales_raw)
        
        print(f"\n缩放统计 (原始值):")
        print(f"  平均缩放: {np.mean(scales_raw, axis=0)}")
        print(f"  最大缩放: {np.max(scales_raw, axis=0)}")
        
        print(f"\n缩放统计 (exp后):")
        print(f"  平均缩放: {np.mean(scales_exp, axis=0)}")
        print(f"  最大缩放: {np.max(scales_exp, axis=0)}")
        
        # 检查异常大的缩放
        large_scale_mask = np.any(scales_exp > 1.0, axis=1)
        if np.any(large_scale_mask):
            print(f"  警告: {np.sum(large_scale_mask):,} 个点有异常大的缩放 (>1.0)")
    
    return plydata

def optimize_opacity_fixed(plydata, method='aggressive'):
    """
    优化不透明度参数（处理Sigmoid前的原始值）
    """
    vertices = plydata['vertex'].data
    opacities_raw = vertices['opacity'].copy()
    opacities_sigmoid = sigmoid(opacities_raw)
    
    print(f"\n优化前不透明度统计 (Sigmoid后):")
    print(f"  平均值: {np.mean(opacities_sigmoid):.6f}")
    print(f"  <0.01的比例: {np.mean(opacities_sigmoid < 0.01):.2%}")
    print(f"  >0.5的比例: {np.mean(opacities_sigmoid > 0.5):.2%}")
    
    # 根据方法选择参数（针对Sigmoid后的值）
    if method == 'aggressive':
        # 激进优化：显著减少不透明区域
        target_mean = 0.15  # 目标平均不透明度
        low_threshold = 0.05  # 低不透明度阈值
        high_threshold = 0.4   # 高不透明度阈值
    elif method == 'moderate':
        # 中等优化：平衡优化
        target_mean = 0.2
        low_threshold = 0.03
        high_threshold = 0.5
    else:  # conservative
        # 保守优化：轻微调整
        target_mean = 0.25
        low_threshold = 0.02
        high_threshold = 0.6
    
    # 计算当前平均值
    current_mean = np.mean(opacities_sigmoid)
    
    # 调整策略：将原始值向目标平均值调整
    if current_mean < target_mean:
        # 当前太透明，增加不透明度
        adjustment = target_mean / current_mean
        new_opacities_raw = opacities_raw * adjustment
    else:
        # 当前太不透明，减少不透明度
        adjustment = current_mean / target_mean
        new_opacities_raw = opacities_raw / adjustment
    
    # 对极端值进行额外处理
    new_opacities_sigmoid = sigmoid(new_opacities_raw)
    
    # 降低过高不透明度的点
    high_mask = new_opacities_sigmoid > high_threshold
    if np.any(high_mask):
        # 对高不透明度点应用更强的减少
        reduction_factor = high_threshold / new_opacities_sigmoid[high_mask]
        new_opacities_raw[high_mask] = new_opacities_raw[high_mask] * reduction_factor
    
    # 增加过低不透明度的点
    low_mask = new_opacities_sigmoid < low_threshold
    if np.any(low_mask):
        # 对低不透明度点应用更强的增加
        boost_factor = low_threshold / new_opacities_sigmoid[low_mask]
        new_opacities_raw[low_mask] = new_opacities_raw[low_mask] * boost_factor
    
    # 最终Sigmoid值
    final_opacities_sigmoid = sigmoid(new_opacities_raw)
    
    print(f"\n优化后不透明度统计 (Sigmoid后):")
    print(f"  平均值: {np.mean(final_opacities_sigmoid):.6f}")
    print(f"  <0.01的比例: {np.mean(final_opacities_sigmoid < 0.01):.2%}")
    print(f"  >0.5的比例: {np.mean(final_opacities_sigmoid > 0.5):.2%}")
    print(f"  变化量: {np.mean(np.abs(final_opacities_sigmoid - opacities_sigmoid)):.6f}")
    
    # 更新数据
    vertices['opacity'] = new_opacities_raw
    return plydata

def optimize_scales_fixed(plydata, max_scale_exp=1.0):
    """优化缩放参数（处理exp前的原始值）"""
    vertices = plydata['vertex'].data
    
    if all(k in vertices.dtype.names for k in ['scale_0', 'scale_1', 'scale_2']):
        scales_raw = np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T
        scales_exp = np.exp(scales_raw)
        
        print(f"\n缩放优化前 (exp后):")
        print(f"  平均缩放: {np.mean(scales_exp, axis=0)}")
        print(f"  最大缩放: {np.max(scales_exp, axis=0)}")
        
        # 限制最大缩放（在exp后的空间）
        scales_exp_clipped = np.clip(scales_exp, 0.001, max_scale_exp)
        
        # 转换回原始值
        scales_raw_new = np.log(scales_exp_clipped)
        
        # 更新数据
        vertices['scale_0'] = scales_raw_new[:, 0]
        vertices['scale_1'] = scales_raw_new[:, 1]
        vertices['scale_2'] = scales_raw_new[:, 2]
        
        print(f"\n缩放优化后 (exp后):")
        print(f"  平均缩放: {np.mean(scales_exp_clipped, axis=0)}")
        print(f"  最大缩放: {np.max(scales_exp_clipped, axis=0)}")
    
    return plydata

def remove_outliers_fixed(plydata, opacity_threshold=0.001, scale_threshold=0.001):
    """移除异常点（使用Sigmoid/exp后的值判断）"""
    vertices = plydata['vertex'].data
    original_count = len(vertices)
    
    # 创建掩码
    keep_mask = np.ones(len(vertices), dtype=bool)
    
    # 1. 移除极低不透明度的点（Sigmoid后）
    if 'opacity' in vertices.dtype.names:
        opacities_sigmoid = sigmoid(vertices['opacity'])
        opacity_mask = opacities_sigmoid > opacity_threshold
        keep_mask = keep_mask & opacity_mask
        print(f"移除极低不透明度点: {np.sum(~opacity_mask):,}")
    
    # 2. 移除极小缩放的点（exp后）
    if all(k in vertices.dtype.names for k in ['scale_0', 'scale_1', 'scale_2']):
        scales_raw = np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T
        scales_exp = np.exp(scales_raw)
        scale_norms = np.linalg.norm(scales_exp, axis=1)
        scale_mask = scale_norms > scale_threshold
        keep_mask = keep_mask & scale_mask
        print(f"移除极小缩放点: {np.sum(~scale_mask):,}")
    
    # 应用过滤
    if np.sum(keep_mask) < original_count:
        print(f"\n过滤前点数: {original_count:,}")
        print(f"过滤后点数: {np.sum(keep_mask):,}")
        print(f"移除点数: {original_count - np.sum(keep_mask):,} ({100*(original_count - np.sum(keep_mask))/original_count:.1f}%)")
        
        # 创建新的顶点数据
        new_vertices = vertices[keep_mask]
        plydata = PlyData([PlyElement.describe(new_vertices, 'vertex')], text=False)
    
    return plydata

def main_fixed():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting 模型后处理优化（修复版）")
    parser.add_argument("input", help="输入PLY文件路径")
    parser.add_argument("--method", choices=['aggressive', 'moderate', 'conservative'], 
                       default='moderate', help="优化强度")
    parser.add_argument("--max-scale", type=float, default=0.5, help="最大缩放限制（exp后）")
    parser.add_argument("--remove-outliers", action="store_true", help="移除异常点")
    parser.add_argument("--opacity-threshold", type=float, default=0.001, help="不透明度阈值（Sigmoid后）")
    parser.add_argument("--scale-threshold", type=float, default=0.001, help="缩放阈值（exp后）")
    parser.add_argument("--output-suffix", default="_optimized_fixed", help="输出文件后缀")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("3D Gaussian Splatting 模型后处理优化（修复版）")
    print("=" * 60)
    
    # 1. 分析原始模型
    plydata = analyze_model(args.input)
    
    # 2. 优化不透明度
    print("\n" + "=" * 60)
    print("步骤1: 优化不透明度参数")
    print("=" * 60)
    plydata = optimize_opacity_fixed(plydata, method=args.method)
    
    # 3. 优化缩放参数
    print("\n" + "=" * 60)
    print("步骤2: 优化缩放参数")
    print("=" * 60)
    plydata = optimize_scales_fixed(plydata, max_scale_exp=args.max_scale)
    
    # 4. 可选：移除异常点
    if args.remove_outliers:
        print("\n" + "=" * 60)
        print("步骤3: 移除异常点")
        print("=" * 60)
        plydata = remove_outliers_fixed(plydata, args.opacity_threshold, args.scale_threshold)
    
    # 5. 保存优化后的模型
    print("\n" + "=" * 60)
    print("保存优化结果")
    print("=" * 60)
    
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.stem}{args.output_suffix}{input_path.suffix}"
    
    print(f"保存优化模型到: {output_path}")
    plydata.write(str(output_path))
    
    # 检查文件大小
    original_size = input_path.stat().st_size / (1024*1024)
    optimized_size = output_path.stat().st_size / (1024*1024)
    
    print(f"原始文件大小: {original_size:.1f} MB")
    print(f"优化文件大小: {optimized_size:.1f} MB")
    print(f"大小变化: {optimized_size - original_size:+.1f} MB")
    
    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
    print(f"原始模型: {args.input}")
    print(f"优化模型: {output_path}")
    print(f"优化强度: {args.method}")
    print("\n使用优化模型:")
    print(f"  ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m {output_path}")
    print(f"  或复制到模型目录:")
    print(f"  cp {output_path} output/tandt_train_20260414_135259/point_cloud/iteration_30000/")
    print(f"  ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/tandt_train_20260414_135259")

if __name__ == "__main__":
    main_fixed()