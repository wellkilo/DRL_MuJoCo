#!/bin/bash
#===============================================================================
# run_webui.sh — 在 Slurm 计算节点上启动 Web UI
#
# 用法:
#   sbatch scripts/slurm/run_webui.sh                          # 1 GPU
#   sbatch --gres=gpu:4 --cpus-per-task=40 scripts/slurm/run_webui.sh  # 4 GPU
#
# 提交后查看 logs/webui_<jobid>.out 获取 SSH 端口转发命令
#===============================================================================

#SBATCH --job-name=drl_webui
#SBATCH --output=logs/webui_%j.out
#SBATCH --error=logs/webui_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --partition=debug

# ======================== 环境配置 ========================

source /nfs/software/miniconda3/bin/activate
conda activate drl_mujoco

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

WEBUI_PORT=8000
MANAGEMENT_NODE="10.179.32.169"    # ← 按实际集群修改

mkdir -p logs output

export RAY_TMPDIR="${PROJECT_DIR}/.ray_tmp_webui_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DISABLE_DASHBOARD=1

NODE_HOSTNAME=$(hostname)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

# ======================== 作业信息 ========================

echo "=============================================="
echo "  DRL MuJoCo — Web UI (Slurm 集群)"
echo "=============================================="
echo "作业ID:     ${SLURM_JOB_ID}"
echo "计算节点:   ${NODE_HOSTNAME}"
echo "GPU数量:    ${NUM_GPUS}"
echo "端口:       ${WEBUI_PORT}"
echo "=============================================="
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║  请在你的【本地电脑终端】执行以下命令:                      ║"
echo "║                                                              ║"
echo "║  ssh -N -L ${WEBUI_PORT}:${NODE_HOSTNAME}:${WEBUI_PORT} username@${MANAGEMENT_NODE}    ║"
echo "║                                                              ║"
echo "║  然后浏览器打开: http://localhost:${WEBUI_PORT}                       ║"
echo "║  (把 username 替换为你的集群用户名)                          ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ======================== 启动 Web UI ========================

echo ">>> 启动 Web UI..."

# 绑定 0.0.0.0, 使 SSH 端口转发可达
python -c "
import uvicorn
uvicorn.run('web.server:app', host='0.0.0.0', port=${WEBUI_PORT}, log_level='info')
"

EXIT_CODE=$?

# ======================== 清理 ========================

rm -rf "${RAY_TMPDIR}" 2>/dev/null
echo "Web UI 已停止, 退出码: ${EXIT_CODE}"
exit ${EXIT_CODE}