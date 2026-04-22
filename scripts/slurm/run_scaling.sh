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
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ======================== 环境配置 ========================

source /nfs/software/miniconda3/bin/activate
conda activate drl_mujoco

# ======================== 参数解析 ========================

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

CONFIG_FILE="${1:-config/config.yaml}"
NUM_GPUS="${2:-1}"

mkdir -p logs
mkdir -p output/scaling

# ======================== Ray 集群配置 ========================

export RAY_TMPDIR="${PROJECT_DIR}/.ray_tmp_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DISABLE_DASHBOARD=1

# 检测节点数
NUM_NODES=$(scontrol show job ${SLURM_JOB_ID} 2>/dev/null | grep -oP 'NumNodes=\K\d+')
NUM_NODES=${NUM_NODES:-1}

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
            srun --nodes=1 --ntasks=1 -w ${worker} \
                ray start --address="${HEAD_NODE}:${RAY_PORT}" \
                --num-cpus=${SLURM_CPUS_PER_TASK} \
                --num-gpus=${GPUS_PER_NODE} \
                --temp-dir="${RAY_TMPDIR}" --block &
            sleep 5
        done
        sleep 10
        ray status
        export RAY_ADDRESS="${HEAD_NODE}:${RAY_PORT}"
    fi
    USE_RAY_CLUSTER=true
else
    USE_RAY_CLUSTER=false
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

[ "${USE_RAY_CLUSTER}" = true ] && ray stop --force 2>/dev/null
rm -rf "${RAY_TMPDIR}" 2>/dev/null

echo "退出码: ${EXIT_CODE}, 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
exit ${EXIT_CODE}