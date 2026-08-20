#!/bin/bash
#===============================================================================
# setup_env.sh — 在 Slurm 集群上创建 Conda 环境 + 前端依赖
#===============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "PROJECT_DIR = ${PROJECT_DIR}"

ENV_NAME="${ENV_NAME:-drl_mujoco}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
NODE_VERSION="${NODE_VERSION:-20}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

# ---------- 加载 conda 初始化脚本 ----------
CONDA_BASE="${CONDA_BASE:-/nfs/software/miniconda3}"
if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "错误: 未找到 ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
echo "conda version: $(conda --version)"

# ---------- 创建 / 复用环境 ----------
if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "环境 ${ENV_NAME} 已存在。"
    read -r -p "是否重新创建？(y/N): " REPLY
    if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
        conda env remove -n "${ENV_NAME}" -y
    fi
fi

if ! conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo ">>> 创建 Conda 环境 (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

# ---------- Python 依赖 ----------
echo ">>> 安装 PyTorch ${PYTORCH_VERSION}: ${PYTORCH_INDEX_URL}"
python -m pip install "torch==${PYTORCH_VERSION}" --index-url "${PYTORCH_INDEX_URL}"

echo ">>> 安装 Python 项目依赖..."
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

# ---------- Node.js ----------
echo ">>> 安装 Node.js ${NODE_VERSION} ..."
conda install -c conda-forge "nodejs=${NODE_VERSION}" -y

# ---------- 验证 ----------
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
echo "Node:       $(node --version)"
echo "npm:        $(npm --version)"

echo ""
echo ">>> 测试 MuJoCo 环境..."
python - <<'PYEOF'
import gymnasium as gym
env = gym.make('Hopper-v5')
obs, info = env.reset()
print(f"Hopper-v5 obs shape: {obs.shape}, action space: {env.action_space}")
env.close()
print("MuJoCo 环境测试通过!")
PYEOF

# ---------- 前端 ----------
echo ""
echo ">>> 构建 Next.js 前端..."
WEB_DIR="${PROJECT_DIR}/web"
if [ ! -d "${WEB_DIR}" ]; then
    echo "警告: 未找到 ${WEB_DIR}, 跳过前端构建。"
else
    pushd "${WEB_DIR}" > /dev/null
    if [ ! -f "package-lock.json" ]; then
        echo "错误: 未找到 ${WEB_DIR}/package-lock.json，无法进行可复现安装。"
        exit 1
    fi
    # 保留 lockfile 和平台可选依赖（Next.js SWC 等），确保集群重建结果一致。
    npm ci
    npm run build
    popd > /dev/null
    echo "前端构建完成 (web/out/)"
fi

echo ""
echo "=============================================="
echo "  环境 ${ENV_NAME} 创建完成!"
echo "  以后开新终端使用方式:"
echo "    conda activate ${ENV_NAME}"
echo "  启动 Web UI: sbatch scripts/slurm/run_webui.sh"
echo "=============================================="
