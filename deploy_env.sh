#!/bin/bash

# ==============================================================================
# Hi_Drum 项目环境自动化部署脚本
# 用途：在新服务器上一键还原 drum 虚拟环境
# ==============================================================================

ENV_NAME="drum"
PYTHON_VERSION="3.10.19"
REQ_FILE="requirements_ubuntu.txt"

echo "🚀 开始配置 $ENV_NAME 虚拟环境..."

# 1. 检查 Conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未检测到 conda，请先安装 Anaconda 或 Miniconda。"
    exit 1
fi

# 2. 检查 requirements 文件是否存在
if [ ! -f "$REQ_FILE" ]; then
    echo "❌ 错误: 找不到 $REQ_FILE 文件，请确保在正确的目录下运行此脚本。"
    exit 1
fi

# 3. 创建 Conda 虚拟环境
echo "📦 正在创建 Conda 环境 ($ENV_NAME) ，使用 Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 4. 激活虚拟环境并安装依赖
# 注意：在 bash 脚本中激活 conda 需要 source conda.sh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "🔄 激活环境 $ENV_NAME..."
conda activate $ENV_NAME

# 升级 pip 以避免旧版本带来的解析错误
echo "⬆️ 升级 pip..."
pip install --upgrade pip

echo "📥 开始安装 requirements_ubuntu.txt 中的依赖（包含精确锁定的包和 CUDA 环境）..."
# 使用清华源加速下载（可选）
pip install -r $REQ_FILE -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "✅ 环境部署完成！"
echo "👉 请运行以下命令激活环境："
echo "    conda activate $ENV_NAME"
