#!/bin/bash
#===============================================================================
# run_distributed.sh — Slurm 提交脚本: 分布式模式训练 (8 Actors)
#
# 用法:
#   sbatch scripts/slurm/run_distributed.sh                    # 默认 Hopper
#   sbatch scripts/slurm/run_distributed.sh config/config.yaml # 指定配置
#===============================================================================

#SBATCH --job-name=drl_dist
#SBATCH --output=logs/dist_%j.out
#SBATCH --error=logs/dist_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --partition=debug

# ======================== 环境配置 ========================

source /nfs/software/miniconda3/bin/activate
conda activate drl_mujoco

# ======================== 变量定义 ========================

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

CONFIG_FILE="${1:-config/config.yaml}"

mkdir -p logs
mkdir -p output

# ======================== Ray 配置 ========================

# 限制 Ray 使用的 CPU 数, 避免超出 Slurm 分配
export RAY_NUM_CPUS=${SLURM_CPUS_PER_TASK:-12}

# Ray 临时目录 (避免写入 /tmp 导致空间或权限问题)
export RAY_TMPDIR="${PROJECT_DIR}/.ray_tmp_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"

# 禁用 Ray Dashboard (集群环境下通常无法访问)
export RAY_DISABLE_DASHBOARD=1

# ======================== 作业信息 ========================

echo "=============================================="
echo "  DRL MuJoCo - 分布式模式训练 (8 Actors)"
echo "=============================================="
echo "作业ID:       ${SLURM_JOB_ID}"
echo "运行节点:     ${SLURM_JOB_NODELIST}"
echo "CPU核心数:    ${SLURM_CPUS_PER_TASK}"
echo "GPU数:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "配置文件:     ${CONFIG_FILE}"
echo "项目目录:     ${PROJECT_DIR}"
echo "Ray临时目录:  ${RAY_TMPDIR}"
echo "开始时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

echo "Python:   $(python --version 2>&1)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "Ray:      $(python -c 'import ray; print(ray.__version__)' 2>&1)"
echo "CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo ""

# ======================== 训练 ========================

echo ">>> 开始训练: python main.py ${CONFIG_FILE}"
echo ""

python main.py "${CONFIG_FILE}"
EXIT_CODE=$?

# ======================== 清理 ========================

if [ -d "${RAY_TMPDIR}" ]; then
    rm -rf "${RAY_TMPDIR}"
    echo "已清理 Ray 临时目录: ${RAY_TMPDIR}"
fi

echo ""
echo "=============================================="
echo "训练结束"
echo "退出码:   ${EXIT_CODE}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

exit ${EXIT_CODE}