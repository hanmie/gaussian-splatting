#!/usr/bin/env python3
"""
3D Gaussian Splatting 模型后处理优化脚本
用于优化PLY模型文件，减少阴影和不透明区域
"""

import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os
from pathlib import Path

def analyze_model(ply_path):
    """分析模型文件"""
    print(f"分析模型: {ply_path}")
    
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex'].data
    
    print(f"高斯点总数: {len(vertices):,}")
    print(f"属性: {vertices.dtype.names}")
    
    # 不透明度分析
    if 'opacity' in vertices.dtype.names:
        opacities = vertices['opacity']
        print(f"\n不透明度统计:")
        print(f"  平均值: {np.mean(opacities):.6f}")
        print(f"  中位数: {np.median(opacities):.6f}")
        print(f"  最小值: {np.min(opacities):.6f}")
        print(f"  最大值: {np.max(opacities):.6f}")
        print(f"  标准差: {np.std(opacities):.6f}")
        
        # 分布统计
        bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
        for low, high in bins:
            count = np.sum((opacities >= low) & (opacities < high))
            percentage = count / len(opacities) * 100
            print(f"  [{low:.2f}-{high:.2f}): {count:,} ({percentage:.1f}%)")
    
    # 缩放分析
    if all(k in vertices.dtype.names for k in ['scale_0', 'scale_1', 'scale_2']):
        scales = np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T
        print(f"\n缩放统计:")
        print(f"  平均缩放 (x,y,z): {np.mean(scales, axis=0)}")
        print(f"  最大缩放 (x,y,z): {np.max(scales, axis=0)}")
        print(f"  缩放标准差: {np.std(scales, axis=0)}")
        
        # 检查异常大的缩放
        large_scale_mask = np.any(scales > 1.0, axis=1)
        if np.any(large_scale_mask):
            print(f"  警告: {np.sum(large_scale_mask):,} 个点有异常大的缩放 (>1.0)")
    
    return plydata

def optimize_opacity(plydata, method='aggressive'):
    """
    优化不透明度参数
    method: 'aggressive' (激进), 'moderate' (中等), 'conservative' (保守)
    """
    vertices = plydata['vertex'].data
    opacities = vertices['opacity'].copy()
    
    print(f"\n优化前不透明度统计:")
    print(f"  平均值: {np.mean(opacities):.6f}")
    print(f"  <0.01的比例: {np.mean(opacities < 0.01):.2%}")
    print(f"  >0.5的比例: {np.mean(opacities > 0.5):.2%}")
    
    # 根据方法选择参数
    if method == 'aggressive':
        # 激进优化：显著减少不透明区域
        low_threshold = 0.05
        high_threshold = 0.3
        low_multiplier = 0.3  # 低不透明度点更透明
        high_multiplier = 0.7  # 高不透明度点降低
        power = 0.8  # 幂次变换
    elif method == 'moderate':
        # 中等优化：平衡优化
        low_threshold = 0.03
        high_threshold = 0.4
        low_multiplier = 0.5
        high_multiplier = 0.8
        power = 0.9
    else:  # conservative
        # 保守优化：轻微调整
        low_threshold = 0.02
        high_threshold = 0.5
        low_multiplier = 0.7
        high_multiplier = 0.9
        power = 0.95
    
    # 应用优化
    new_opacities = opacities.copy()
    
    # 1. 对低不透明度点：使其更透明（减少阴影）
    low_mask = opacities < low_threshold
    new_opacities[low_mask] = opacities[low_mask] * low_multiplier
    
    # 2. 对高不透明度点：适当降低（减少不透明斑块）
    high_mask = opacities > high_threshold
    new_opacities[high_mask] = opacities[high_mask] * high_multiplier
    
    # 3. 应用幂次变换平滑分布
    new_opacities = np.power(new_opacities, power)
    
    # 4. 确保在有效范围内
    new_opacities = np.clip(new_opacities, 0.001, 0.99)
    
    print(f"\n优化后不透明度统计:")
    print(f"  平均值: {np.mean(new_opacities):.6f}")
    print(f"  <0.01的比例: {np.mean(new_opacities < 0.01):.2%}")
    print(f"  >0.5的比例: {np.mean(new_opacities > 0.5):.2%}")
    print(f"  变化量: {np.mean(np.abs(new_opacities - opacities)):.6f}")
    
    # 更新数据
    vertices['opacity'] = new_opacities
    return plydata

def optimize_scales(plydata, max_scale=1.0):
    """优化缩放参数，限制异常大的缩放"""
    vertices = plydata['vertex'].data
    
    if all(k in vertices.dtype.names for k in ['scale_0', 'scale_1', 'scale_2']):
        scales = np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T
        
        print(f"\n缩放优化前:")
        print(f"  平均缩放: {np.mean(scales, axis=0)}")
        print(f"  最大缩放: {np.max(scales, axis=0)}")
        
        # 限制最大缩放
        scales = np.clip(scales, 0.001, max_scale)
        
        # 更新数据
        vertices['scale_0'] = scales[:, 0]
        vertices['scale_1'] = scales[:, 1]
        vertices['scale_2'] = scales[:, 2]
        
        print(f"\n缩放优化后:")
        print(f"  平均缩放: {np.mean(scales, axis=0)}")
        print(f"  最大缩放: {np.max(scales, axis=0)}")
    
    return plydata

def optimize_colors(plydata, saturation_boost=1.1):
    """优化颜色参数，增强饱和度"""
    vertices = plydata['vertex'].data
    
    # 检查颜色属性（球谐函数系数）
    color_props = ['f_dc_0', 'f_dc_1', 'f_dc_2']
    if all(k in vertices.dtype.names for k in color_props):
        colors = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']]).T
        
        print(f"\n颜色优化前:")
        print(f"  平均颜色: {np.mean(colors, axis=0)}")
        print(f"  颜色标准差: {np.std(colors, axis=0)}")
        
        # 增强饱和度
        colors = colors * saturation_boost
        
        # 限制在合理范围内
        colors = np.clip(colors, -1.0, 1.0)
        
        # 更新数据
        vertices['f_dc_0'] = colors[:, 0]
        vertices['f_dc_1'] = colors[:, 1]
        vertices['f_dc_2'] = colors[:, 2]
        
        print(f"\n颜色优化后:")
        print(f"  平均颜色: {np.mean(colors, axis=0)}")
        print(f"  颜色标准差: {np.std(colors, axis=0)}")
    
    return plydata

def remove_outliers(plydata, opacity_threshold=0.001, scale_threshold=0.001):
    """移除异常点"""
    vertices = plydata['vertex'].data
    original_count = len(vertices)
    
    # 创建掩码
    keep_mask = np.ones(len(vertices), dtype=bool)
    
    # 1. 移除极低不透明度的点
    if 'opacity' in vertices.dtype.names:
        opacity_mask = vertices['opacity'] > opacity_threshold
        keep_mask = keep_mask & opacity_mask
        print(f"移除极低不透明度点: {np.sum(~opacity_mask):,}")
    
    # 2. 移除极小缩放的点
    if all(k in vertices.dtype.names for k in ['scale_0', 'scale_1', 'scale_2']):
        scales = np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T
        scale_norms = np.linalg.norm(scales, axis=1)
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

def save_optimized_model(plydata, input_path, suffix="_optimized"):
    """保存优化后的模型"""
    input_path = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
    
    print(f"\n保存优化模型到: {output_path}")
    plydata.write(str(output_path))
    
    # 检查文件大小
    original_size = input_path.stat().st_size / (1024*1024)
    optimized_size = output_path.stat().st_size / (1024*1024)
    
    print(f"原始文件大小: {original_size:.1f} MB")
    print(f"优化文件大小: {optimized_size:.1f} MB")
    print(f"大小变化: {optimized_size - original_size:+.1f} MB")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting 模型后处理优化")
    parser.add_argument("input", help="输入PLY文件路径")
    parser.add_argument("--method", choices=['aggressive', 'moderate', 'conservative'], 
                       default='moderate', help="优化强度")
    parser.add_argument("--max-scale", type=float, default=1.0, help="最大缩放限制")
    parser.add_argument("--saturation", type=float, default=1.1, help="颜色饱和度增强")
    parser.add_argument("--remove-outliers", action="store_true", help="移除异常点")
    parser.add_argument("--opacity-threshold", type=float, default=0.001, help="不透明度阈值")
    parser.add_argument("--scale-threshold", type=float, default=0.001, help="缩放阈值")
    parser.add_argument("--output-suffix", default="_optimized", help="输出文件后缀")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("3D Gaussian Splatting 模型后处理优化")
    print("=" * 60)
    
    # 1. 分析原始模型
    plydata = analyze_model(args.input)
    
    # 2. 优化不透明度（主要解决阴影和不透明问题）
    print("\n" + "=" * 60)
    print("步骤1: 优化不透明度参数")
    print("=" * 60)
    plydata = optimize_opacity(plydata, method=args.method)
    
    # 3. 优化缩放参数
    print("\n" + "=" * 60)
    print("步骤2: 优化缩放参数")
    print("=" * 60)
    plydata = optimize_scales(plydata, max_scale=args.max_scale)
    
    # 4. 优化颜色参数
    print("\n" + "=" * 60)
    print("步骤3: 优化颜色参数")
    print("=" * 60)
    plydata = optimize_colors(plydata, saturation_boost=args.saturation)
    
    # 5. 可选：移除异常点
    if args.remove_outliers:
        print("\n" + "=" * 60)
        print("步骤4: 移除异常点")
        print("=" * 60)
        plydata = remove_outliers(plydata, args.opacity_threshold, args.scale_threshold)
    
    # 6. 保存优化后的模型
    print("\n" + "=" * 60)
    print("保存优化结果")
    print("=" * 60)
    output_path = save_optimized_model(plydata, args.input, args.output_suffix)
    
    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
    print(f"原始模型: {args.input}")
    print(f"优化模型: {output_path}")
    print(f"优化强度: {args.method}")
    print("\n使用优化模型:")
    print(f"  ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m {output_path}")
    print(f"  python render.py -m {output_path.parent.parent} --white_background")

if __name__ == "__main__":
    main()