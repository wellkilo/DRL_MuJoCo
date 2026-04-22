#!/bin/bash
#===============================================================================
# setup_env.sh — 在 Slurm 集群上创建项目 Conda 环境
#
# 用法 (在管理节点上直接运行, 不需要 sbatch):
#   bash scripts/slurm/setup_env.sh
#
# 或者通过 Slurm 交互式节点运行:
#   srun --partition=debug --cpus-per-task=4 --mem=8G --pty bash -i
#   bash scripts/slurm/setup_env.sh
#===============================================================================

set -e

ENV_NAME="drl_mujoco"
PYTHON_VERSION="3.9"

echo "=============================================="
echo "  创建 Conda 环境: ${ENV_NAME}"
echo "=============================================="

# 加载 Conda
source /nfs/software/miniconda3/bin/activate

# 检查环境是否已存在
if conda info --envs | grep -q "${ENV_NAME}"; then
    echo "环境 ${ENV_NAME} 已存在。"
    read -p "是否重新创建？(y/N): " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "跳过创建, 仅更新依赖..."
        conda activate ${ENV_NAME}
        pip install -r requirements.txt
        echo "依赖更新完成！"
        exit 0
    fi
fi

# 创建新环境
echo ""
echo ">>> 创建 Conda 环境 (Python ${PYTHON_VERSION})..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# 激活环境
conda activate ${ENV_NAME}

# 安装 PyTorch (根据集群 CUDA 版本选择)
echo ""
echo ">>> 安装 PyTorch (CUDA)..."
# 如果集群是 CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 如果集群是 CUDA 12.1, 改用:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
echo ""
echo ">>> 安装项目依赖..."
pip install -r requirements.txt

# 验证安装
echo ""
echo "=============================================="
echo "  环境验证"
echo "=============================================="
echo "Python:     $(python --version)"
echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Ray:        $(python -c 'import ray; print(ray.__version__)')"
echo "Gymnasium:  $(python -c 'import gymnasium; print(gymnasium.__version__)')"
echo "MuJoCo:     $(python -c 'import mujoco; print(mujoco.__version__)')"

# MuJoCo 渲染测试 (headless)
echo ""
echo ">>> 测试 MuJoCo 环境创建..."
python -c "
import gymnasium as gym
env = gym.make('Hopper-v5')
obs, info = env.reset()
print(f'Hopper-v5 obs shape: {obs.shape}, action space: {env.action_space}')
env.close()
print('MuJoCo 环境测试通过!')
"

# ======================== 构建 Next.js 前端 ========================

echo ""
echo ">>> 构建 Next.js 前端..."
if command -v npm &> /dev/null; then
    cd "${PROJECT_DIR}/web"
    [ ! -d "node_modules" ] && npm install
    npm run build
    cd "${PROJECT_DIR}"
    echo "前端构建完成 (web/out/)"
else
    echo "npm 不可用, 请在本地执行以下命令后上传:"
    echo "  cd web && npm install && npm run build"
    echo "然后将 web/out/ 目录上传到集群"
fi

echo ""
echo "=============================================="
echo "  环境 ${ENV_NAME} 创建完成!"
echo "  使用方式: conda activate ${ENV_NAME}"
echo "  启动 Web UI: sbatch scripts/slurm/run_webui.sh"
echo "=============================================="