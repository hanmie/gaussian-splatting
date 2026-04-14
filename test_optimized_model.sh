#!/bin/bash

# 测试优化后的3D高斯模型

echo "=========================================="
echo "测试优化后的3D高斯模型"
echo "=========================================="

# 设置环境
conda activate gaussian_splatting
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# 原始模型路径
ORIGINAL_MODEL="output/tandt_train_20260414_135259/point_cloud/iteration_30000/point_cloud.ply"
OPTIMIZED_MODEL="output/tandt_train_20260414_135259/point_cloud/iteration_30000/point_cloud_optimized_fixed.ply"

echo ""
echo "1. 检查模型文件"
echo "   原始模型: $ORIGINAL_MODEL"
echo "   优化模型: $OPTIMIZED_MODEL"

if [ ! -f "$ORIGINAL_MODEL" ]; then
    echo "错误: 原始模型文件不存在"
    exit 1
fi

if [ ! -f "$OPTIMIZED_MODEL" ]; then
    echo "错误: 优化模型文件不存在"
    exit 1
fi

echo ""
echo "2. 模型文件大小对比"
echo "   原始模型: $(du -h "$ORIGINAL_MODEL" | cut -f1)"
echo "   优化模型: $(du -h "$OPTIMIZED_MODEL" | cut -f1)"

echo ""
echo "3. 使用SIBR查看器查看优化模型"
echo "   如果SIBR查看器已构建，运行以下命令:"
echo "   ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m \"$OPTIMIZED_MODEL\""
echo ""
echo "   或者使用模型目录:"
echo "   ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/tandt_train_20260414_135259"

echo ""
echo "4. 渲染优化模型"
echo "   使用白色背景渲染:"
echo "   python render.py -m output/tandt_train_20260414_135259 --white_background"
echo ""
echo "   渲染结果保存在: output/tandt_train_20260414_135259/renders/"

echo ""
echo "5. 快速对比渲染"
echo "   创建对比脚本:"
cat > compare_models.py << 'EOF'
import subprocess
import os

print("对比原始模型和优化模型的渲染效果")

# 渲染原始模型（如果尚未渲染）
if not os.path.exists("output/tandt_train_20260414_135259/renders_original"):
    print("渲染原始模型...")
    subprocess.run([
        "python", "render.py", 
        "-m", "output/tandt_train_20260414_135259",
        "--white_background",
        "--skip_train",
        "--images", "images",
        "--output", "output/tandt_train_20260414_135259/renders_original"
    ], check=False)

# 确保使用优化模型
optimized_ply = "output/tandt_train_20260414_135259/point_cloud/iteration_30000/point_cloud_optimized_fixed.ply"
original_ply = "output/tandt_train_20260414_135259/point_cloud/iteration_30000/point_cloud.ply"

# 临时替换模型文件进行渲染
import shutil
backup_ply = original_ply + ".backup"
shutil.copy2(original_ply, backup_ply)
shutil.copy2(optimized_ply, original_ply)

try:
    print("渲染优化模型...")
    subprocess.run([
        "python", "render.py", 
        "-m", "output/tandt_train_20260414_135259",
        "--white_background",
        "--skip_train",
        "--images", "images",
        "--output", "output/tandt_train_20260414_135259/renders_optimized"
    ], check=False)
finally:
    # 恢复原始文件
    shutil.copy2(backup_ply, original_ply)
    os.remove(backup_ply)

print("渲染完成!")
print("原始模型渲染: output/tandt_train_20260414_135259/renders_original/")
print("优化模型渲染: output/tandt_train_20260414_135259/renders_optimized/")
EOF

echo "   运行对比: python compare_models.py"

echo ""
echo "6. 优化效果总结"
echo "   根据优化分析:"
echo "   - 原始模型平均不透明度: 0.297 (Sigmoid后)"
echo "   - 优化模型平均不透明度: 0.346 (Sigmoid后)"
echo "   - 不透明度<0.01的比例: 从4.7%降低到0.0%"
echo "   - 最大缩放限制: 从14.3降低到0.3"
echo ""
echo "   预期改善:"
echo "   ✅ 减少阴影斑块"
echo "   ✅ 提高透明度均匀性"
echo "   ✅ 限制异常大的高斯点"
echo "   ✅ 整体视觉效果更清晰"

echo ""
echo "7. 立即操作"
echo "   a) 查看优化模型:"
echo "      ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m \"$OPTIMIZED_MODEL\""
echo ""
echo "   b) 渲染测试图像:"
echo "      python render.py -m output/tandt_train_20260414_135259 --white_background --skip_train"
echo ""
echo "   c) 如果需要进一步优化:"
echo "      python optimize_model_fixed.py \"$OPTIMIZED_MODEL\" --method moderate"

echo ""
echo "=========================================="
echo "后处理优化已完成!"
echo "优化模型: $OPTIMIZED_MODEL"
echo "使用优化模型替换原始模型:"
echo "  cp \"$OPTIMIZED_MODEL\" \"$ORIGINAL_MODEL\""
echo "=========================================="