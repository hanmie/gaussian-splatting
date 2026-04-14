#!/usr/bin/env python3
"""
3D Gaussian Splatting 不透明点过滤脚本
专门针对不透明区域进行强过滤
"""

import numpy as np
from plyfile import PlyData, PlyElement
import argparse
from pathlib import Path

def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def analyze_opacity_distribution(ply_path):
    """详细分析不透明度分布"""
    print(f"分析模型: {ply_path}")
    
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex'].data
    
    if 'opacity' not in vertices.dtype.names:
        print("错误: 模型没有不透明度属性")
        return None
    
    opacities_raw = vertices['opacity']
    opacities_sigmoid = sigmoid(opacities_raw)
    
    print(f"高斯点总数: {len(vertices):,}")
    
    # 详细分布统计
    print(f"\n不透明度详细分布 (Sigmoid后):")
    
    bins = [
        (0.0, 0.001), (0.001, 0.005), (0.005, 0.01),
        (0.01, 0.05), (0.05, 0.1), (0.1, 0.2),
        (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
        (0.8, 0.9), (0.9, 1.0)
    ]
    
    for low, high in bins:
        count = np.sum((opacities_sigmoid >= low) & (opacities_sigmoid < high))
        percentage = count / len(opacities_sigmoid) * 100
        if count > 0:
            print(f"  [{low:.3f}-{high:.3f}): {count:,} ({percentage:.2f}%)")
    
    # 关键阈值统计
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"\n关键阈值统计:")
    for threshold in thresholds:
        count_above = np.sum(opacities_sigmoid >= threshold)
        percentage_above = count_above / len(opacities_sigmoid) * 100
        print(f"  ≥{threshold:.2f}: {count_above:,} ({percentage_above:.2f}%)")
    
    return plydata, opacities_sigmoid

def filter_by_opacity(plydata, opacity_threshold=0.3, keep_ratio=0.7, method='aggressive'):
    """
    根据不透明度过滤点
    opacity_threshold: 高于此阈值的点被视为"不透明点"
    keep_ratio: 保留的不透明点比例
    method: 'aggressive'/'moderate'/'conservative'
    """
    vertices = plydata['vertex'].data
    opacities_sigmoid = sigmoid(vertices['opacity'])
    
    original_count = len(vertices)
    print(f"\n原始点数: {original_count:,}")
    print(f"平均不透明度: {np.mean(opacities_sigmoid):.4f}")
    
    # 识别不透明点
    opaque_mask = opacities_sigmoid >= opacity_threshold
    opaque_count = np.sum(opaque_mask)
    transparent_count = original_count - opaque_count
    
    print(f"不透明点(≥{opacity_threshold}): {opaque_count:,} ({opaque_count/original_count*100:.1f}%)")
    print(f"透明点(<{opacity_threshold}): {transparent_count:,} ({transparent_count/original_count*100:.1f}%)")
    
    # 根据方法确定保留比例
    if method == 'aggressive':
        # 激进：只保留少量不透明点
        target_opaque_ratio = 0.1  # 目标不透明点比例
        target_keep_ratio = 0.3    # 总体保留比例
    elif method == 'moderate':
        target_opaque_ratio = 0.2
        target_keep_ratio = 0.5
    else:  # conservative
        target_opaque_ratio = 0.3
        target_keep_ratio = 0.7
    
    # 计算需要保留的不透明点数量
    target_opaque_count = int(original_count * target_opaque_ratio)
    
    if opaque_count > target_opaque_count:
        # 需要减少不透明点
        # 按不透明度排序，保留最高的
        opaque_indices = np.where(opaque_mask)[0]
        opaque_opacities = opacities_sigmoid[opaque_indices]
        
        # 按不透明度降序排序
        sorted_indices = np.argsort(-opaque_opacities)
        keep_opaque_indices = opaque_indices[sorted_indices[:target_opaque_count]]
        
        # 创建新的掩码
        new_opaque_mask = np.zeros(original_count, dtype=bool)
        new_opaque_mask[keep_opaque_indices] = True
        
        print(f"减少不透明点: {opaque_count:,} → {target_opaque_count:,}")
    else:
        new_opaque_mask = opaque_mask.copy()
    
    # 处理透明点
    transparent_mask = ~opaque_mask
    transparent_indices = np.where(transparent_mask)[0]
    
    # 计算需要保留的透明点数量
    target_total_count = int(original_count * target_keep_ratio)
    target_transparent_count = target_total_count - np.sum(new_opaque_mask)
    
    if len(transparent_indices) > target_transparent_count:
        # 随机选择透明点保留
        np.random.seed(42)  # 固定随机种子以便复现
        keep_transparent_indices = np.random.choice(
            transparent_indices, 
            size=target_transparent_count, 
            replace=False
        )
        
        new_transparent_mask = np.zeros(original_count, dtype=bool)
        new_transparent_mask[keep_transparent_indices] = True
    else:
        new_transparent_mask = transparent_mask.copy()
    
    # 合并掩码
    keep_mask = new_opaque_mask | new_transparent_mask
    removed_count = original_count - np.sum(keep_mask)
    
    print(f"\n过滤结果:")
    print(f"  原始点数: {original_count:,}")
    print(f"  保留点数: {np.sum(keep_mask):,}")
    print(f"  移除点数: {removed_count:,} ({removed_count/original_count*100:.1f}%)")
    print(f"  保留的不透明点: {np.sum(new_opaque_mask):,}")
    print(f"  保留的透明点: {np.sum(new_transparent_mask):,}")
    
    # 应用过滤
    if np.sum(keep_mask) < original_count:
        new_vertices = vertices[keep_mask]
        plydata = PlyData([PlyElement.describe(new_vertices, 'vertex')], text=False)
        
        # 分析过滤后的不透明度
        filtered_opacities = sigmoid(new_vertices['opacity'])
        print(f"\n过滤后统计:")
        print(f"  平均不透明度: {np.mean(filtered_opacities):.4f}")
        print(f"  不透明度≥0.3: {np.mean(filtered_opacities >= 0.3):.2%}")
        print(f"  不透明度≥0.5: {np.mean(filtered_opacities >= 0.5):.2%}")
    
    return plydata

def reduce_opacity_values(plydata, reduction_factor=0.7, high_opacity_threshold=0.5):
    """降低高不透明度点的值"""
    vertices = plydata['vertex'].data
    opacities_raw = vertices['opacity'].copy()
    opacities_sigmoid = sigmoid(opacities_raw)
    
    print(f"\n降低高不透明度值:")
    print(f"  降低前平均不透明度: {np.mean(opacities_sigmoid):.4f}")
    
    # 识别高不透明度点
    high_opacity_mask = opacities_sigmoid >= high_opacity_threshold
    high_count = np.sum(high_opacity_mask)
    
    if high_count > 0:
        print(f"  高不透明度点(≥{high_opacity_threshold}): {high_count:,}")
        
        # 降低高不透明度点的原始值
        # 将Sigmoid后的值乘以reduction_factor，然后转换回原始值
        high_opacities_sigmoid = opacities_sigmoid[high_opacity_mask]
        reduced_opacities_sigmoid = high_opacities_sigmoid * reduction_factor
        
        # 将Sigmoid后的值转换回原始值
        # op = sigmoid(x) => x = log(op / (1 - op))
        reduced_opacities_raw = np.log(reduced_opacities_sigmoid / (1 - reduced_opacities_sigmoid))
        
        # 更新原始值
        opacities_raw[high_opacity_mask] = reduced_opacities_raw
        
        # 更新数据
        vertices['opacity'] = opacities_raw
        
        # 验证
        new_opacities_sigmoid = sigmoid(opacities_raw)
        print(f"  降低后平均不透明度: {np.mean(new_opacities_sigmoid):.4f}")
        print(f"  高不透明度点减少: {np.mean(new_opacities_sigmoid >= high_opacity_threshold):.2%}")
    
    return plydata

def main():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting 不透明点过滤")
    parser.add_argument("input", help="输入PLY文件路径")
    parser.add_argument("--opacity-threshold", type=float, default=0.3, 
                       help="不透明度阈值，高于此值视为不透明点")
    parser.add_argument("--method", choices=['aggressive', 'moderate', 'conservative'], 
                       default='aggressive', help="过滤强度")
    parser.add_argument("--reduce-opacity", action="store_true", 
                       help="降低高不透明度点的值")
    parser.add_argument("--reduction-factor", type=float, default=0.6,
                       help="高不透明度点降低因子")
    parser.add_argument("--high-opacity-threshold", type=float, default=0.5,
                       help="高不透明度阈值")
    parser.add_argument("--output-suffix", default="_filtered", help="输出文件后缀")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("3D Gaussian Splatting 不透明点强过滤")
    print("=" * 60)
    
    # 1. 分析模型
    result = analyze_opacity_distribution(args.input)
    if result is None:
        return
    
    plydata, opacities_sigmoid = result
    
    # 2. 过滤不透明点
    print("\n" + "=" * 60)
    print("步骤1: 过滤不透明点")
    print("=" * 60)
    plydata = filter_by_opacity(
        plydata, 
        opacity_threshold=args.opacity_threshold,
        method=args.method
    )
    
    # 3. 可选：降低高不透明度值
    if args.reduce_opacity:
        print("\n" + "=" * 60)
        print("步骤2: 降低高不透明度值")
        print("=" * 60)
        plydata = reduce_opacity_values(
            plydata,
            reduction_factor=args.reduction_factor,
            high_opacity_threshold=args.high_opacity_threshold
        )
    
    # 4. 保存结果
    print("\n" + "=" * 60)
    print("保存过滤结果")
    print("=" * 60)
    
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.stem}{args.output_suffix}{input_path.suffix}"
    
    print(f"保存过滤模型到: {output_path}")
    plydata.write(str(output_path))
    
    # 检查文件大小
    original_size = input_path.stat().st_size / (1024*1024)
    filtered_size = output_path.stat().st_size / (1024*1024)
    
    print(f"原始文件大小: {original_size:.1f} MB")
    print(f"过滤文件大小: {filtered_size:.1f} MB")
    print(f"大小变化: {filtered_size - original_size:+.1f} MB")
    
    print("\n" + "=" * 60)
    print("过滤完成!")
    print("=" * 60)
    print(f"原始模型: {args.input}")
    print(f"过滤模型: {output_path}")
    print(f"过滤强度: {args.method}")
    print(f"不透明度阈值: {args.opacity_threshold}")
    
    print("\n使用过滤模型:")
    print(f"  ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m {output_path}")
    print(f"  或替换原始文件:")
    print(f"  cp {output_path} {args.input}")
    print(f"  ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m {input_path.parent.parent}")

if __name__ == "__main__":
    main()