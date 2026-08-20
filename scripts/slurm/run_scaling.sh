#!/bin/bash
#===============================================================================
# run_scaling.sh — Slurm 提交脚本: GPU 扩展性实验
# 支持多 GPU 多节点, Ray 自动调度 Actor 到各 GPU 所在节点
#
# 用法:
#   sbatch --gres=gpu:4 --cpus-per-task=40 --mem=64G \
#       scripts/slurm/run_scaling.sh config/scaling/hopper_gpu4.yaml 4
#===============================================================================

#SBATCH --job-name=drl_scale
#SBATCH --output=logs/scale_%j.out
#SBATCH --error=logs/scale_%j.err
#SBATCH --partition=debug
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -uo pipefail

# ======================== 环境配置 ========================

CONDA_BASE="${CONDA_BASE:-/nfs/software/miniconda3}"
[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ] || { echo "[FATAL] 未找到 Conda: ${CONDA_BASE}"; exit 1; }
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate drl_mujoco || exit 1

# ======================== 参数解析 ========================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_DIR}" || exit 1
[ -f "main.py" ] || { echo "[FATAL] 请在项目根目录提交 sbatch 作业"; exit 1; }

CONFIG_FILE="${1:-config/config.yaml}"
NUM_GPUS="${2:-1}"

mkdir -p logs
mkdir -p output/scaling

# ======================== Ray 集群配置 ========================

RAY_TMP_ROOT="${PROJECT_DIR}/.ray_tmp_${SLURM_JOB_ID}"
export RAY_TMPDIR="${RAY_TMP_ROOT}/$(hostname -s)"
mkdir -p "${RAY_TMPDIR}"
export RAY_DISABLE_DASHBOARD=1
USE_RAY_CLUSTER=false
cleanup() {
    if [ "${USE_RAY_CLUSTER:-false}" = true ]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
    if [[ -n "${RAY_TMP_ROOT:-}" && "${RAY_TMP_ROOT}" == "${PROJECT_DIR}/.ray_tmp_"* ]]; then
        rm -rf -- "${RAY_TMP_ROOT}"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 检测节点数
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"

if [ "${NUM_NODES}" -gt 1 ]; then
    echo ">>> 多节点模式: ${NUM_NODES} 节点"
    HEAD_NODE=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -1)
    RAY_PORT=6379
    GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)

    if [ "$(hostname)" = "${HEAD_NODE}" ]; then
        ray start --head --port=${RAY_PORT} \
            --num-cpus=${SLURM_CPUS_PER_TASK} \
            --num-gpus=${GPUS_PER_NODE} \
            --temp-dir="${RAY_TMPDIR}" --block &
        sleep 10

        for worker in $(scontrol show hostnames ${SLURM_JOB_NODELIST} | tail -n +2); do
            WORKER_TMPDIR="${RAY_TMP_ROOT}/${worker}"
            mkdir -p "${WORKER_TMPDIR}"
            srun --nodes=1 --ntasks=1 -w ${worker} \
                ray start --address="${HEAD_NODE}:${RAY_PORT}" \
                --num-cpus=${SLURM_CPUS_PER_TASK} \
                --num-gpus=${GPUS_PER_NODE} \
                --temp-dir="${WORKER_TMPDIR}" --block &
            sleep 5
        done
        sleep 10
        ray status
        export RAY_ADDRESS="${HEAD_NODE}:${RAY_PORT}"
    fi
    USE_RAY_CLUSTER=true
fi

# ======================== 作业信息 ========================

echo "=============================================="
echo "  DRL MuJoCo - GPU 扩展性实验"
echo "=============================================="
echo "作业ID:       ${SLURM_JOB_ID}"
echo "运行节点:     ${SLURM_JOB_NODELIST}"
echo "节点数:       ${NUM_NODES}"
echo "CPU核心/节点: ${SLURM_CPUS_PER_TASK}"
echo "GPU数(申请):  ${NUM_GPUS}"
echo "配置文件:     ${CONFIG_FILE}"
echo "开始时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
echo ""
echo "PyTorch CUDA GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>&1)"

# ======================== 训练 ========================

if [ "${USE_RAY_CLUSTER}" = true ]; then
    RAY_ADDRESS="${RAY_ADDRESS}" python main.py "${CONFIG_FILE}"
else
    python main.py "${CONFIG_FILE}"
fi
EXIT_CODE=$?

# ======================== 清理 ========================

echo "退出码: ${EXIT_CODE}, 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
exit ${EXIT_CODE}
