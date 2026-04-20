#!/bin/bash
#===============================================================================
# run_plot.sh — 训练完成后绘图 (无需 GPU, 低资源)
#
# 用法:
#   sbatch scripts/slurm/run_plot.sh               # 绘制全部图表
#   sbatch scripts/slurm/run_plot.sh --training     # 仅绘制训练详情图
#   sbatch scripts/slurm/run_plot.sh --comparison   # 仅绘制总览对比图
#===============================================================================

#SBATCH --job-name=drl_plot
#SBATCH --output=logs/plot_%j.out
#SBATCH --error=logs/plot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-01:00:00
#SBATCH --partition=debug

source /nfs/software/miniconda3/bin/activate
conda activate drl_mujoco

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

PLOT_MODE="${1:---all}"
mkdir -p output/figures

echo "=============================================="
echo "  DRL MuJoCo - 绘图任务"
echo "=============================================="
echo "作业ID:   ${SLURM_JOB_ID}"
echo "绘图模式: ${PLOT_MODE}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

if [ "${PLOT_MODE}" = "--all" ] || [ "${PLOT_MODE}" = "--comparison" ]; then
    echo ""
    echo ">>> 绘制多环境总览对比图..."
    python scripts/plot_comparison.py --output_dir output/figures
fi

if [ "${PLOT_MODE}" = "--all" ] || [ "${PLOT_MODE}" = "--training" ]; then
    echo ""
    echo ">>> 绘制各环境训练详情图..."
    python scripts/plot_training.py --all --output_dir output/figures
fi

echo ""
echo "=============================================="
echo "绘图完成！输出目录: output/figures/"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
ls -lh output/figures/