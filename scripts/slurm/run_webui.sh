#!/bin/bash
#===============================================================================
# run_webui.sh — 在 Slurm 计算节点上启动 DRL MuJoCo Web UI
#
# 提交方式(必须在项目根目录下提交,不能在别处):
#   cd ~/DRL_MuJoCo
#   sbatch scripts/slurm/run_webui.sh
#===============================================================================

#SBATCH --job-name=drl_webui
#SBATCH --output=logs/webui_%j.log
#SBATCH --error=logs/webui_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --partition=debug

set -uo pipefail

# ======================== 1. 激活 conda 环境 ========================

CONDA_BASE="${CONDA_BASE:-/nfs/software/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "${CONDA_BASE}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/bin/activate"
else
    echo "[FATAL] 未找到 Conda: ${CONDA_BASE}"
    exit 1
fi
conda activate drl_mujoco || exit 1

# ======================== 2. 项目根目录(关键修复) ========================
# sbatch 把脚本拷贝到 /var/spool/slurm/...,$0/BASH_SOURCE 不指向源文件,
# 必须用 SLURM_SUBMIT_DIR(你 sbatch 时所在的目录)。

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd "${PROJECT_DIR}" || { echo "[FATAL] 进不去 ${PROJECT_DIR}"; exit 1; }

# 健壮性自检
if [ ! -f "${PROJECT_DIR}/web/package.json" ]; then
    echo "[FATAL] PROJECT_DIR 解析错了: ${PROJECT_DIR}"
    echo "        没找到 ${PROJECT_DIR}/web/package.json"
    echo "        请确保你在项目根目录下执行 sbatch(cd ~/DRL_MuJoCo 再 sbatch)"
    exit 1
fi

# ======================== 3. 基础变量 ========================

WEBUI_PORT="${WEBUI_PORT:-8000}"
MANAGEMENT_NODE="${MANAGEMENT_NODE:-node1}"   # ← 你的登录节点;按需改
SSH_USER="${SSH_USER:-${USER:-$(id -un)}}"

mkdir -p logs output

# 让 uvicorn 自身日志、server.py 内的 print、子进程 stdout 都实时落盘
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

SLURM_JOB_ID="${SLURM_JOB_ID:-local}"
export RAY_TMPDIR="${PROJECT_DIR}/.ray_tmp_webui_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
cleanup() {
    if [[ -n "${RAY_TMPDIR:-}" && "${RAY_TMPDIR}" == "${PROJECT_DIR}/.ray_tmp_webui_"* ]]; then
        rm -rf -- "${RAY_TMPDIR}"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export RAY_DISABLE_DASHBOARD=1
export WEBUI_HOST=0.0.0.0
export WEBUI_PORT="${WEBUI_PORT}"

# ======================== 4. 节点信息(关键修复) ========================
# 你的 /etc/hosts 里写了 127.0.0.1 super-01 之类自环行,
# 用 getent 会拿到 127.0.0.1。必须用 hostname -I 拿真实 IP。

NODE_HOSTNAME="$(hostname -s)"
NODE_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^127\.' | grep -v '^$' | head -1)"
NODE_IP="${NODE_IP:-${NODE_HOSTNAME}}"
NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"

echo "=============================================="
echo "  DRL MuJoCo — Web UI (Slurm 集群)"
echo "=============================================="
echo "作业ID:       ${SLURM_JOB_ID}"
echo "计算节点名:   ${NODE_HOSTNAME}"
echo "计算节点IP:   ${NODE_IP}"
echo "GPU数量:      ${NUM_GPUS}"
echo "监听端口:     ${WEBUI_PORT}"
echo "项目目录:     ${PROJECT_DIR}"
echo "Python:       $(python --version 2>&1)"
echo "开始时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
echo ""

# ======================== 5. 检查 web/out/index.html ========================
# 你已确认 web/out/index.html 存在,这里只做存在性检查,不再触发 npm install/build。
# 如果将来不存在了,会给出清晰提示让你手动构建。

if [ ! -f "${PROJECT_DIR}/web/out/index.html" ]; then
    echo "[ERROR] 未找到 ${PROJECT_DIR}/web/out/index.html"
    echo "        前端静态产物不存在,请先在登录节点手动构建一次:"
    echo "          conda activate drl_mujoco"
    echo "          cd ${PROJECT_DIR}/web"
    echo "          NEXT_EXPORT=1 npm run build"
    echo "        构建完成后再重新 sbatch 本脚本。"
    exit 1
fi
echo ">>> 已确认 web/out/index.html 存在,跳过前端构建。"

# ======================== 6. 端口转发指引 ========================

cat <<EOF

==============================================================
  请在【你本地的笔记本】终端(不是 node1!)执行下面命令做端口转发:
--------------------------------------------------------------
  ★ 首选(用计算节点 IP, 跨子网最稳):
    ssh -N -L ${WEBUI_PORT}:${NODE_IP}:${WEBUI_PORT} ${SSH_USER}@${MANAGEMENT_NODE}

  ☆ 备选(用计算节点短名,需登录节点能解析 ${NODE_HOSTNAME}):
    ssh -N -L ${WEBUI_PORT}:${NODE_HOSTNAME}:${WEBUI_PORT} ${SSH_USER}@${MANAGEMENT_NODE}

  说明:
    - 命令在本地电脑(笔记本)执行,不要在 node1 上执行
    - 命令会一直挂着不返回提示符(-N),保持不动别 Ctrl+C
    - 转发起来后浏览器打开:  http://localhost:${WEBUI_PORT}
==============================================================

EOF

# ======================== 7. 启动 uvicorn ========================

echo ">>> 启动 FastAPI(uvicorn host=0.0.0.0 port=${WEBUI_PORT})..."
echo ""

UVICORN_CMD=(python -u -m uvicorn web.server:app --host 0.0.0.0 --port "${WEBUI_PORT}" --log-level info)
if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "${UVICORN_CMD[@]}"
else
    "${UVICORN_CMD[@]}"
fi

EXIT_CODE=$?

# ======================== 8. 清理 ========================

echo ""
echo "=============================================="
echo "Web UI 已停止"
echo "退出码:   ${EXIT_CODE}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
exit ${EXIT_CODE}
