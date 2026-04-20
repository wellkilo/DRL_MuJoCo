#!/bin/bash
#===============================================================================
# run_single.sh — Slurm 提交脚本: 单机模式训练 (1 Actor)
#
# 用法:
#   sbatch scripts/slurm/run_single.sh                            # 默认 Hopper
#   sbatch scripts/slurm/run_single.sh config/config_single.yaml  # 指定配置
#===============================================================================

#SBATCH --job-name=drl_single
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --partition=debug

# ======================== 环境配置 ========================

source /nfs/software/miniconda3/bin/activate
conda activate drl_mujoco

# ======================== 变量定义 ========================

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

CONFIG_FILE="${1:-config/config_single.yaml}"

mkdir -p logs
mkdir -p output

# ======================== 作业信息 ========================

echo "=============================================="
echo "  DRL MuJoCo - 单机模式训练"
echo "=============================================="
echo "作业ID:       ${SLURM_JOB_ID}"
echo "运行节点:     ${SLURM_JOB_NODELIST}"
echo "CPU核心数:    ${SLURM_CPUS_PER_TASK}"
echo "GPU数:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "配置文件:     ${CONFIG_FILE}"
echo "项目目录:     ${PROJECT_DIR}"
echo "开始时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

echo "Python:   $(python --version 2>&1)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "Ray:      $(python -c 'import ray; print(ray.__version__)' 2>/dev/null)"
echo "CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo ""

# ======================== 训练 ========================

echo ">>> 开始训练: python main.py ${CONFIG_FILE}"
echo ""

python main.py "${CONFIG_FILE}"
EXIT_CODE=$?

# ======================== 结束 ========================

echo ""
echo "=============================================="
echo "训练结束"
echo "退出码:   ${EXIT_CODE}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

exit ${EXIT_CODE}