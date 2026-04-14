#!/bin/bash

# 3D Gaussian Splatting 环境设置与测试脚本
# 作者：自动生成
# 日期：2026年4月14日

set -e  # 遇到错误时退出脚本

echo "=========================================="
echo "3D Gaussian Splatting 环境设置与测试脚本"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否在项目根目录
check_project_root() {
    if [ ! -f "train.py" ] || [ ! -d "submodules" ]; then
        print_error "请在 3D Gaussian Splatting 项目根目录运行此脚本"
        exit 1
    fi
    print_success "项目根目录检查通过"
}

# 检查 Conda 环境
check_conda_env() {
    print_info "检查 Conda 环境..."
    
    if ! command -v conda &> /dev/null; then
        print_error "未找到 Conda，请先安装 Conda"
        exit 1
    fi
    
    # 检查是否在 gaussian_splatting 环境中
    if [[ "$CONDA_DEFAULT_ENV" != "gaussian_splatting" ]]; then
        print_warning "不在 gaussian_splatting 环境中，尝试激活..."
        
        if conda activate gaussian_splatting 2>/dev/null; then
            print_success "已激活 gaussian_splatting 环境"
        else
            print_error "无法激活 gaussian_splatting 环境，请先创建："
            echo "conda env create -f environment.yml"
            exit 1
        fi
    else
        print_success "已在 gaussian_splatting 环境中"
    fi
}

# 设置编译器环境变量
setup_compiler() {
    print_info "设置编译器环境变量..."
    
    # 检查系统 GCC
    if [ ! -f "/usr/bin/gcc" ]; then
        print_error "未找到 /usr/bin/gcc，请安装系统 GCC"
        exit 1
    fi
    
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export CUDAHOSTCXX=/usr/bin/g++
    
    print_success "编译器环境变量已设置："
    echo "  CC=$CC"
    echo "  CXX=$CXX"
    echo "  CUDAHOSTCXX=$CUDAHOSTCXX"
}

# 安装 CUDA 扩展模块
install_cuda_extensions() {
    print_info "开始安装 CUDA 扩展模块..."
    
    # 检查 CUDA 是否可用
    print_info "检查 CUDA 可用性..."
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
    
    # 安装 diff-gaussian-rasterization
    print_info "安装 diff-gaussian-rasterization..."
    cd submodules/diff-gaussian-rasterization
    pip install . --no-build-isolation
    cd ../..
    
    # 安装 simple-knn
    print_info "安装 simple-knn..."
    cd submodules/simple-knn
    pip install . --no-build-isolation
    cd ../..
    
    # 安装 fused-ssim
    print_info "安装 fused-ssim..."
    cd submodules/fused-ssim
    pip install . --no-build-isolation
    cd ../..
    
    print_success "CUDA 扩展模块安装完成"
}

# 验证安装
verify_installation() {
    print_info "验证安装..."
    
    # 测试导入 CUDA 扩展
    print_info "测试导入 CUDA 扩展模块..."
    if python -c "import simple_knn; import diff_gaussian_rasterization; import fused_ssim; print('所有CUDA扩展模块导入成功！')"; then
        print_success "CUDA 扩展模块导入测试通过"
    else
        print_error "CUDA 扩展模块导入失败"
        exit 1
    fi
    
    # 测试导入主项目模块
    print_info "测试导入主项目模块..."
    if python -c "import sys; sys.path.insert(0, '.'); from gaussian_renderer import render; print('gaussian_renderer 导入成功！')"; then
        print_success "主项目模块导入测试通过"
    else
        print_error "主项目模块导入失败"
        exit 1
    fi
}

# 快速测试训练
quick_test_training() {
    print_info "开始快速测试训练..."
    
    # 检查示例数据是否存在
    if [ ! -d "assets/tandt_db/tandt/train" ]; then
        print_warning "示例数据不存在，跳过测试训练"
        return 0
    fi
    
    TEST_OUTPUT="test_output_$(date +%Y%m%d_%H%M%S)"
    print_info "测试输出目录: $TEST_OUTPUT"
    
    # 运行简短训练
    print_info "运行简短训练（100次迭代）..."
    python train.py -s assets/tandt_db/tandt/train \
        --iterations 100 \
        --model_path "$TEST_OUTPUT" \
        --quiet
    
    if [ $? -eq 0 ] && [ -d "$TEST_OUTPUT" ]; then
        print_success "测试训练完成！"
        echo "输出目录: $TEST_OUTPUT"
        echo "内容:"
        ls -la "$TEST_OUTPUT/"
    else
        print_warning "测试训练可能未完全成功，但环境基本可用"
    fi
}

# 显示使用指南
show_usage_guide() {
    echo ""
    echo "=========================================="
    echo "环境设置完成！以下是一些常用命令："
    echo "=========================================="
    echo ""
    echo "1. 完整训练（使用示例数据）："
    echo "   python train.py -s assets/tandt_db/tandt/train"
    echo ""
    echo "2. 带评估的训练："
    echo "   python train.py -s assets/tandt_db/tandt/train --eval"
    echo "   python render.py -m output/<模型目录>"
    echo "   python metrics.py -m output/<模型目录>"
    echo ""
    echo "3. 处理自己的图像："
    echo "   python convert.py -s my_dataset --resize"
    echo ""
    echo "4. 构建可视化查看器："
    echo "   cd SIBR_viewers"
    echo "   cmake -Bbuild ."
    echo "   cmake --build build --target install --config RelWithDebInfo"
    echo ""
    echo "5. 实时训练可视化："
    echo "   # 终端1: python train.py -s <数据集> --ip 127.0.0.1 --port 6009"
    echo "   # 终端2: ./SIBR_viewers/install/bin/SIBR_remoteGaussian_app"
    echo ""
    echo "更多详细信息请查看 '使用指南.md' 文件"
    echo "=========================================="
}

# 永久设置编译器环境变量（可选）
setup_permanent() {
    read -p "是否永久设置编译器环境变量到 ~/.bashrc？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo 'export CC=/usr/bin/gcc' >> ~/.bashrc
        echo 'export CXX=/usr/bin/g++' >> ~/.bashrc
        echo 'export CUDAHOSTCXX=/usr/bin/g++' >> ~/.bashrc
        print_success "已永久设置编译器环境变量，请运行 'source ~/.bashrc' 或重新打开终端"
    fi
}

# 主函数
main() {
    echo ""
    print_info "开始 3D Gaussian Splatting 环境设置..."
    
    # 执行各个步骤
    check_project_root
    check_conda_env
    setup_compiler
    install_cuda_extensions
    verify_installation
    quick_test_training
    setup_permanent
    show_usage_guide
    
    echo ""
    print_success "环境设置与测试完成！"
    print_info "现在可以开始使用 3D Gaussian Splatting 了。"
}

# 运行主函数
main "$@"