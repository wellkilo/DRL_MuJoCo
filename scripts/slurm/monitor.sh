#!/bin/bash
#===============================================================================
# monitor.sh — 实用工具: 监控正在运行的训练作业
#
# 用法:
#   bash scripts/slurm/monitor.sh             # 查看所有自己的作业
#   bash scripts/slurm/monitor.sh <job_id>    # 实时查看指定作业日志
#   bash scripts/slurm/monitor.sh --cancel-all # 取消所有自己的作业
#===============================================================================

if [ "$1" = "--cancel-all" ]; then
    echo ">>> 取消所有作业..."
    scancel -u "$USER"
    echo "已发送取消请求。"
    squeue -u "$USER"
    exit 0
fi

if [ -n "$1" ] && [ "$1" != "--cancel-all" ]; then
    JOB_ID=$1
    echo ">>> 实时查看作业 ${JOB_ID} 的输出日志..."
    echo "    (按 Ctrl+C 停止)"
    echo ""

    LOG_FILE=$(ls -t logs/*_${JOB_ID}.out 2>/dev/null | head -1)
    if [ -z "${LOG_FILE}" ]; then
        echo "未找到日志文件, 等待生成..."
        for i in $(seq 1 30); do
            sleep 2
            LOG_FILE=$(ls -t logs/*_${JOB_ID}.out 2>/dev/null | head -1)
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