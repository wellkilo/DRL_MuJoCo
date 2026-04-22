#!/bin/bash
#===============================================================================
# run_gpu_experiments.sh — 一键提交 GPU 扩展性对比实验
#
# 用法:
#   bash scripts/slurm/run_gpu_experiments.sh               # 提交全部
#   bash scripts/slurm/run_gpu_experiments.sh --dry-run      # 仅预览
#   bash scripts/slurm/run_gpu_experiments.sh --envs hopper  # 仅 Hopper
#   bash scripts/slurm/run_gpu_experiments.sh --gpus 4 8     # 仅 4 和 8 GPU
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DRY_RUN=false
ENVS=("hopper" "walker2d" "halfcheetah")
GPU_COUNTS=(4 8 16 32)

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --envs)     shift; ENVS=(); while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do ENVS+=("$1"); shift; done ;;
        --gpus)     shift; GPU_COUNTS=(); while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do GPU_COUNTS+=("$1"); shift; done ;;
        *) shift ;;
    esac
done

# ========= 集群参数 (请根据实际集群调整!) =========
GPUS_PER_NODE=4          # 每节点最多 GPU 数
CPUS_PER_GPU=10          # 每 GPU 分配的 CPU 核心
MEM_PER_GPU=16           # 每 GPU 分配的内存 (GB)

echo "=============================================="
echo "  DRL MuJoCo — GPU 扩展性对比实验"
echo "  环境: ${ENVS[*]}  |  GPU: ${GPU_COUNTS[*]}"
echo "=============================================="

# 确保配置已生成
[ ! -d "${PROJECT_DIR}/config/scaling" ] && \
    (cd "${PROJECT_DIR}" && python scripts/gen_scaling_configs.py --gpu_counts ${GPU_COUNTS[@]})

mkdir -p "${PROJECT_DIR}/logs"
SUBMITTED=0

for env in "${ENVS[@]}"; do
    echo "--- ${env} ---"
    for num_gpus in "${GPU_COUNTS[@]}"; do
        CFG="config/scaling/${env}_gpu${num_gpus}.yaml"
        [ ! -f "${PROJECT_DIR}/${CFG}" ] && echo "  [跳过] ${CFG}" && continue

        NODES=$(( (num_gpus + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
        GPUS_NODE=$(( num_gpus < GPUS_PER_NODE ? num_gpus : GPUS_PER_NODE ))
        CPUS=$(( GPUS_NODE * CPUS_PER_GPU ))
        MEM=$(( GPUS_NODE * MEM_PER_GPU ))

        CMD="sbatch --job-name=drl_${env}_g${num_gpus} \
            --nodes=${NODES} --gres=gpu:${GPUS_NODE} \
            --cpus-per-task=${CPUS} --mem=${MEM}G \
            ${SCRIPT_DIR}/run_scaling.sh ${CFG} ${num_gpus}"

        echo "  [GPU=${num_gpus}] nodes=${NODES} gpus/node=${GPUS_NODE} cpus=${CPUS} mem=${MEM}G"
        if [ "$DRY_RUN" = false ]; then
            OUTPUT=$(cd "${PROJECT_DIR}" && eval ${CMD})
            echo "    → ${OUTPUT}"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    echo ""
done

echo "已提交 ${SUBMITTED} 个作业"
echo "查看状态: squeue -u \$USER"
echo "分析结果: python scripts/analyze_scaling.py --input_dir output/scaling"