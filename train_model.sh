#!/bin/bash

# 3D Gaussian Splatting 模型训练脚本
# 用法: ./train_model.sh [数据集路径] [模型名称] [选项]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认值
DEFAULT_DATA_PATH="assets/tandt_db/tandt/train"
DEFAULT_MODEL_NAME="gaussian_model"
DEFAULT_ITERATIONS=30000
DEFAULT_RESOLUTION=1

# 打印帮助信息
print_help() {
    echo "3D Gaussian Splatting 模型训练脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -d, --data PATH      数据集路径 (默认: $DEFAULT_DATA_PATH)"
    echo "  -m, --model NAME     模型名称 (默认: $DEFAULT_MODEL_NAME)"
    echo "  -i, --iterations N   训练迭代次数 (默认: $DEFAULT_ITERATIONS)"
    echo "  -r, --resolution N   图像分辨率 (1=原始, 2=一半, 4=四分之一) (默认: $DEFAULT_RESOLUTION)"
    echo "  -e, --eval           使用训练/测试分割进行评估"
    echo "  -w, --white-bg       使用白色背景"
    echo "  -a, --accelerated    使用加速优化器 (sparse_adam)"
    echo "  -aa, --antialiasing  启用抗锯齿"
    echo "  -g, --gpu MEM        指定GPU内存限制 (GB)"
    echo "  -h, --help           显示此帮助信息"
    echo ""
    echo "本工程示例:"
    echo "  1. 快速测试 (100次迭代):"
    echo "     $0 -d assets/tandt_db/tandt/train -i 100"
    echo ""
    echo "  2. 完整训练 Tanks&Temples train 场景:"
    echo "     $0 -d assets/tandt_db/tandt/train -m tandt_train --eval"
    echo ""
    echo "  3. 训练 Tanks&Temples truck 场景:"
    echo "     $0 -d assets/tandt_db/tandt/truck -m tandt_truck -i 30000 --eval"
    echo ""
    echo "  4. 训练 Deep Blending drjohnson 场景:"
    echo "     $0 -d assets/tandt_db/db/drjohnson -m db_drjohnson --eval --white-bg"
    echo ""
    echo "  5. 使用加速优化器训练:"
    echo "     $0 -d assets/tandt_db/tandt/train -m accelerated --accelerated"
    echo ""
    echo "  6. 低VRAM配置训练 (8GB GPU):"
    echo "     $0 -d assets/tandt_db/tandt/train -m low_vram -g 8 -i 15000"
    echo ""
    echo "通用示例:"
    echo "  $0 -d my_dataset -m my_model -i 10000"
    echo "  $0 --data assets/tandt_db/tandt/train --model test --eval"
    echo "  $0 --eval --accelerated --antialiasing"
}

# 解析命令行参数
parse_args() {
    DATA_PATH=$DEFAULT_DATA_PATH
    MODEL_NAME=$DEFAULT_MODEL_NAME
    ITERATIONS=$DEFAULT_ITERATIONS
    RESOLUTION=$DEFAULT_RESOLUTION
    EVAL=false
    WHITE_BG=false
    ACCELERATED=false
    ANTIALIASING=false
    GPU_MEM=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--data)
                DATA_PATH="$2"
                shift 2
                ;;
            -m|--model)
                MODEL_NAME="$2"
                shift 2
                ;;
            -i|--iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            -r|--resolution)
                RESOLUTION="$2"
                shift 2
                ;;
            -e|--eval)
                EVAL=true
                shift
                ;;
            -w|--white-bg)
                WHITE_BG=true
                shift
                ;;
            -a|--accelerated)
                ACCELERATED=true
                shift
                ;;
            -aa|--antialiasing)
                ANTIALIASING=true
                shift
                ;;
            -g|--gpu)
                GPU_MEM="$2"
                shift 2
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

# 检查环境
check_environment() {
    echo -e "${BLUE}[1/6] 检查环境...${NC}"
    
    # 检查是否在正确的conda环境中
    if [[ "$CONDA_DEFAULT_ENV" != "gaussian_splatting" ]]; then
        echo -e "${RED}错误: 请在 gaussian_splatting conda 环境中运行${NC}"
        echo "运行: conda activate gaussian_splatting"
        exit 1
    fi
    
    # 设置编译器环境变量
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export CUDAHOSTCXX=/usr/bin/g++
    
    # 检查CUDA扩展
    if ! python -c "import diff_gaussian_rasterization" &>/dev/null; then
        echo -e "${YELLOW}警告: diff_gaussian_rasterization 未安装，尝试安装...${NC}"
        cd submodules/diff-gaussian-rasterization
        pip install . --no-build-isolation
        cd ../..
    fi
    
    echo -e "${GREEN}✓ 环境检查通过${NC}"
}

# 检查数据集
check_dataset() {
    echo -e "${BLUE}[2/6] 检查数据集...${NC}"
    
    if [ ! -d "$DATA_PATH" ]; then
        echo -e "${RED}错误: 数据集路径不存在: $DATA_PATH${NC}"
        exit 1
    fi
    
    # 检查数据集结构
    if [ ! -d "$DATA_PATH/images" ]; then
        echo -e "${YELLOW}警告: 未找到 images/ 目录，数据集可能格式不正确${NC}"
    fi
    
    if [ ! -d "$DATA_PATH/sparse" ]; then
        echo -e "${YELLOW}警告: 未找到 sparse/ 目录，数据集可能格式不正确${NC}"
    fi
    
    echo -e "${GREEN}✓ 数据集: $DATA_PATH${NC}"
    echo "  图像数量: $(find "$DATA_PATH/images" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)"
    echo "  稀疏文件: $(find "$DATA_PATH/sparse" -name "*.bin" 2>/dev/null | wc -l)"
}

# 准备训练参数
prepare_training() {
    echo -e "${BLUE}[3/6] 准备训练参数...${NC}"
    
    # 创建输出目录
    OUTPUT_DIR="output/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    
    # 构建训练命令
    TRAIN_CMD="python train.py -s \"$DATA_PATH\" -m \"$OUTPUT_DIR\" --iterations $ITERATIONS -r $RESOLUTION"
    
    # 添加可选参数
    if [ "$EVAL" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --eval"
    fi
    
    if [ "$WHITE_BG" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --white_background"
    fi
    
    if [ "$ACCELERATED" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --optimizer_type sparse_adam"
    fi
    
    if [ "$ANTIALIASING" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --antialiasing"
    fi
    
    # GPU内存优化
    if [ -n "$GPU_MEM" ]; then
        if [ "$GPU_MEM" -lt 8 ]; then
            TRAIN_CMD="$TRAIN_CMD --data_device cpu"
            TRAIN_CMD="$TRAIN_CMD --densify_grad_threshold 0.001"
            echo -e "${YELLOW}注意: GPU内存较小 ($GPU_MEM GB)，启用CPU数据存储和更高的梯度阈值${NC}"
        elif [ "$GPU_MEM" -lt 16 ]; then
            TRAIN_CMD="$TRAIN_CMD --densify_grad_threshold 0.0005"
            echo -e "${YELLOW}注意: GPU内存中等 ($GPU_MEM GB)，启用中等梯度阈值${NC}"
        fi
    fi
    
    # 保存训练配置
    cat > "$OUTPUT_DIR/training_config.txt" << EOF
训练配置:
==========
数据集: $DATA_PATH
模型目录: $OUTPUT_DIR
迭代次数: $ITERATIONS
分辨率: $RESOLUTION
评估模式: $EVAL
白色背景: $WHITE_BG
加速优化: $ACCELERATED
抗锯齿: $ANTIALIASING
GPU内存: ${GPU_MEM:-自动}
开始时间: $(date)
命令: $TRAIN_CMD
EOF
    
    echo -e "${GREEN}✓ 训练参数准备完成${NC}"
    echo "  输出目录: $OUTPUT_DIR"
    echo "  迭代次数: $ITERATIONS"
    echo "  评估模式: $EVAL"
}

# 执行训练
run_training() {
    echo -e "${BLUE}[4/6] 开始训练...${NC}"
    echo "命令: $TRAIN_CMD"
    echo ""
    
    # 开始训练
    eval $TRAIN_CMD
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 训练完成${NC}"
    else
        echo -e "${RED}✗ 训练失败${NC}"
        exit 1
    fi
}

# 后处理
post_training() {
    echo -e "${BLUE}[5/6] 训练后处理...${NC}"
    
    # 检查生成的模型文件
    MODEL_FILES=$(find "$OUTPUT_DIR" -name "*.ply" -o -name "*.pth" | wc -l)
    if [ "$MODEL_FILES" -eq 0 ]; then
        echo -e "${YELLOW}警告: 未找到模型文件${NC}"
    else
        echo -e "${GREEN}✓ 找到 $MODEL_FILES 个模型文件${NC}"
    fi
    
    # 如果有评估模式，运行渲染和评估
    if [ "$EVAL" = true ]; then
        echo "运行评估..."
        python render.py -m "$OUTPUT_DIR" --skip_train
        python metrics.py -m "$OUTPUT_DIR"
    fi
    
    # 计算训练时间
    END_TIME=$(date +%s)
    TRAINING_TIME=$((END_TIME - START_TIME))
    echo "训练用时: $(date -u -d @$TRAINING_TIME +'%H小时%M分钟%S秒')"
}

# 生成报告
generate_report() {
    echo -e "${BLUE}[6/6] 生成训练报告...${NC}"
    
    REPORT_FILE="$OUTPUT_DIR/training_report.md"
    
    cat > "$REPORT_FILE" << EOF
# 3D Gaussian Splatting 训练报告

## 训练信息
- **模型名称**: $MODEL_NAME
- **训练时间**: $(date)
- **训练用时**: $(date -u -d @$TRAINING_TIME +'%H小时%M分钟%S秒')
- **输出目录**: $OUTPUT_DIR

## 训练配置
- **数据集**: $DATA_PATH
- **迭代次数**: $ITERATIONS
- **分辨率**: $RESOLUTION
- **评估模式**: $EVAL
- **白色背景**: $WHITE_BG
- **加速优化**: $ACCELERATED
- **抗锯齿**: $ANTIALIASING

## 模型文件
\`\`\`
$(find "$OUTPUT_DIR" -type f -name "*.ply" -o -name "*.pth" -o -name "*.msgpack" | sort)
\`\`\`

## 文件统计
\`\`\`
$(ls -la "$OUTPUT_DIR/")
\`\`\`

## 下一步操作

### 1. 可视化模型
\`\`\`bash
cd SIBR_viewers
./install/bin/SIBR_gaussianViewer_app -m "$OUTPUT_DIR"
\`\`\`

### 2. 渲染图像
\`\`\`bash
python render.py -m "$OUTPUT_DIR"
\`\`\`

### 3. 计算指标
\`\`\`bash
python metrics.py -m "$OUTPUT_DIR"
\`\`\`

### 4. 继续训练
\`\`\`bash
python train.py -s "$DATA_PATH" -m "$OUTPUT_DIR" --start_checkpoint "$OUTPUT_DIR/chkpnt30000.pth" --iterations 60000
\`\`\`

---

*报告生成时间: $(date)*
EOF
    
    echo -e "${GREEN}✓ 训练报告已生成: $REPORT_FILE${NC}"
}

# 主函数
main() {
    START_TIME=$(date +%s)
    
    echo "=========================================="
    echo "    3D Gaussian Splatting 模型训练"
    echo "=========================================="
    echo ""
    
    parse_args "$@"
    check_environment
    check_dataset
    prepare_training
    run_training
    post_training
    generate_report
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}训练完成！${NC}"
    echo "模型保存在: $OUTPUT_DIR"
    echo "查看报告: cat $OUTPUT_DIR/training_report.md"
    echo "=========================================="
}

# 运行主函数
main "$@"