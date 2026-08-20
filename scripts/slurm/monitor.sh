#!/bin/bash
#===============================================================================
# monitor.sh — 实用工具: 监控正在运行的训练作业
#
# 用法:
#   bash scripts/slurm/monitor.sh             # 查看所有自己的作业
#   bash scripts/slurm/monitor.sh <job_id>    # 实时查看指定作业日志
#   bash scripts/slurm/monitor.sh --cancel-all # 取消所有自己的作业
#===============================================================================

set -u

ARGUMENT="${1:-}"

if [ "${ARGUMENT}" = "--cancel-all" ]; then
    echo ">>> 取消所有作业..."
    scancel -u "$USER"
    echo "已发送取消请求。"
    squeue -u "$USER"
    exit 0
fi

if [ -n "${ARGUMENT}" ] && [ "${ARGUMENT}" != "--cancel-all" ]; then
    JOB_ID="${ARGUMENT}"
    echo ">>> 实时查看作业 ${JOB_ID} 的输出日志..."
    echo "    (按 Ctrl+C 停止)"
    echo ""

    find_log_file() {
        local candidate
        for candidate in logs/*_"${JOB_ID}".log logs/*_"${JOB_ID}".out; do
            [ -f "${candidate}" ] && printf '%s\n' "${candidate}" && return 0
        done
        return 1
    }

    LOG_FILE="$(find_log_file || true)"
    if [ -z "${LOG_FILE}" ]; then
        echo "未找到日志文件, 等待生成..."
        for i in $(seq 1 30); do
            sleep 2
            LOG_FILE="$(find_log_file || true)"
            [ -n "${LOG_FILE}" ] && break
            echo "  等待中... (${i}/30)"
        done
    fi

    if [ -n "${LOG_FILE}" ]; then
        echo "日志文件: ${LOG_FILE}"
        echo "=============================================="
        tail -f "${LOG_FILE}"
    else
        echo "错误: 始终未找到日志文件。"
        exit 1
    fi
else
    echo "=============================================="
    echo "  当前用户 ($USER) 的作业列表"
    echo "=============================================="
    squeue -u "$USER" -o "%.10i %.15j %.8T %.10M %.6D %.4C %.6m %R"

    echo ""
    echo "=============================================="
    echo "  集群节点状态"
    echo "=============================================="
    sinfo -o "%.15N %.6D %.10P %.11T %.6c %.8m %.8G"

    echo ""
    echo "用法提示:"
    echo "  查看作业日志: bash scripts/slurm/monitor.sh <job_id>"
    echo "  取消所有作业: bash scripts/slurm/monitor.sh --cancel-all"
fi
