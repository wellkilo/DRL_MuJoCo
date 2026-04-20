#!/bin/bash
#===============================================================================
# run_all_envs.sh — 批量提交全部 3 环境 × 2 模式 = 6 个训练任务
#
# 用法:
#   bash scripts/slurm/run_all_envs.sh           # 提交全部 6 个任务
#   bash scripts/slurm/run_all_envs.sh --dry-run  # 仅打印命令, 不实际提交
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "[DRY-RUN 模式] 仅打印命令, 不实际提交"
    echo ""
fi

# ======================== 任务定义 ========================

TASKS=(
    "Hopper|config/config.yaml|config/config_single.yaml|drl_hopper_dist|drl_hopper_single"
    "Walker2d|config/walker2d.yaml|config/walker2d_single.yaml|drl_walker_dist|drl_walker_single"
    "HalfCheetah|config/halfcheetah.yaml|config/halfcheetah_single.yaml|drl_cheetah_dist|drl_cheetah_single"
)

echo "=============================================="
echo "  DRL MuJoCo — 批量提交训练任务"
echo "  项目目录: ${PROJECT_DIR}"
echo "  共计 ${#TASKS[@]} 个环境 × 2 模式 = $((${#TASKS[@]} * 2)) 个作业"
echo "=============================================="
echo ""

mkdir -p "${PROJECT_DIR}/logs"

SUBMITTED=0

for task in "${TASKS[@]}"; do
    IFS='|' read -r env_name dist_config single_config \
                    dist_jobname single_jobname <<< "$task"

    echo "--- ${env_name} ---"

    # 分布式模式
    CMD_DIST="sbatch --job-name=${dist_jobname} \
        ${SCRIPT_DIR}/run_distributed.sh ${dist_config}"
    echo "  [分布式] ${CMD_DIST}"
    if [ "$DRY_RUN" = false ]; then
        OUTPUT=$(cd "${PROJECT_DIR}" && ${CMD_DIST})
        echo "  → ${OUTPUT}"
        SUBMITTED=$((SUBMITTED + 1))
    fi

    # 单机模式
    CMD_SINGLE="sbatch --job-name=${single_jobname} \
        ${SCRIPT_DIR}/run_single.sh ${single_config}"
    echo "  [单机]   ${CMD_SINGLE}"
    if [ "$DRY_RUN" = false ]; then
        OUTPUT=$(cd "${PROJECT_DIR}" && ${CMD_SINGLE})
        echo "  → ${OUTPUT}"
        SUBMITTED=$((SUBMITTED + 1))
    fi

    echo ""
done

if [ "$DRY_RUN" = false ]; then
    echo "=============================================="
    echo "  已提交 ${SUBMITTED} 个作业"
    echo "  查看作业状态: squeue -u \$USER"
    echo "  取消所有作业: scancel -u \$USER"
    echo "=============================================="
fi