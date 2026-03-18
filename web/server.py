# DRL MuJoCo Web UI 服务器
#
# 提供 Web 界面实时监控分布式训练过程
# 功能：启动/停止训练、实时可视化训练指标、对比单机与分布式性能、视频演示

from __future__ import annotations

import asyncio
import csv
import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Union

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from drl.video_generator import generate_comparison_videos

# 初始化 FastAPI 应用
app = FastAPI(title="DRL MuJoCo Web UI")

# 项目路径配置
REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)  # 确保输出目录存在

# 静态资源和模板配置
app.mount("/static", StaticFiles(directory=REPO_ROOT / "web/static"), name="static")
templates = Jinja2Templates(directory=REPO_ROOT / "web/templates")

# 全局变量
training_task: asyncio.subprocess.Process | None = None  # 当前训练进程
clients: list[WebSocket] = []  # WebSocket 客户端列表
video_generation_process: asyncio.subprocess.Process | None = None  # 视频生成进程
video_generation_status: dict[str, Any] = {"status": "idle"}  # 视频生成状态


@app.get("/")
async def get_index() -> FileResponse:
    """返回主页"""
    return FileResponse(REPO_ROOT / "web/templates/index.html")


def load_metrics_file(metrics_path: Path) -> list[dict[str, Any]]:
    """
    从 CSV 文件加载训练指标
    Args:
        metrics_path: 指标文件路径
    Returns:
        指标数据列表（每个元素为一行的数据）
    """
    if not metrics_path.exists():
        return []
    metrics: list[dict[str, Any]] = []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        # 转换数据类型：步数和回合数为整数，其他为浮点数
        for row in reader:
            metrics.append({k: float(v) if k not in ("step", "episodes") else int(float(v)) for k, v in row.items()})
    return metrics


# ==================== API 端点 ====================

@app.get("/api/metrics/distributed")
async def get_metrics_distributed() -> list[dict[str, Any]]:
    """获取分布式训练指标"""
    return load_metrics_file(OUTPUT_DIR / "metrics.csv")


@app.get("/api/metrics/single")
async def get_metrics_single() -> list[dict[str, Any]]:
    """获取单机训练指标"""
    return load_metrics_file(OUTPUT_DIR / "metrics_single.csv")


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket) -> None:
    """
    WebSocket 端点：实时推送训练指标
    客户端连接后会定期接收到最新的训练数据
    """
    await websocket.accept()
    clients.append(websocket)
    try:
        # 保持连接活跃
        while True:
            await asyncio.sleep(0.1)
    except Exception:
        pass
    finally:
        # 断开连接时移除客户端
        clients.remove(websocket)


@app.post("/api/training/distributed/start")
async def start_training_distributed() -> dict[str, str]:
    """启动分布式训练（8个Actor）"""
    global training_task
    if training_task and training_task.returncode is None:
        return {"status": "already running"}

    # 创建子进程运行训练
    training_task = await asyncio.create_subprocess_exec(
        "python", str(REPO_ROOT / "main.py"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,  # 合并标准输出和错误输出
        cwd=str(REPO_ROOT),
    )

    # 启动监控任务
    asyncio.create_task(monitor_training())
    return {"status": "started"}


@app.post("/api/training/single/start")
async def start_training_single() -> dict[str, str]:
    """启动单机训练（使用单机配置文件）"""
    global training_task
    if training_task and training_task.returncode is None:
        return {"status": "already running"}

    # 创建子进程运行单机训练
    training_task = await asyncio.create_subprocess_exec(
        "python", str(REPO_ROOT / "main.py"), str(REPO_ROOT / "config" / "config_single.yaml"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    # 启动监控任务
    asyncio.create_task(monitor_training())
    return {"status": "started"}


@app.post("/api/training/stop")
async def stop_training() -> dict[str, str]:
    """停止训练进程"""
    global training_task
    if training_task and training_task.returncode is None:
        print("[Server] Stopping training, giving time to save model...", flush=True)
        # 先尝试优雅终止（SIGTERM）
        training_task.terminate()
        try:
            # 等待 15 秒让进程正常退出并保存模型
            await asyncio.wait_for(training_task.wait(), timeout=15.0)
            print("[Server] Training process exited gracefully", flush=True)
        except asyncio.TimeoutError:
            print("[Server] Timeout, forcing training process to stop...", flush=True)
            # 超时则强制终止（SIGKILL）
            training_task.kill()
            await training_task.wait()
    return {"status": "stopped"}


# ==================== 监控任务 ====================

async def monitor_training() -> None:
    """
    监控训练进程并实时推送指标
    定期读取训练指标文件，通过 WebSocket 推送给所有连接的客户端
    """
    if not training_task:
        return

    # 当训练进程运行时持续监控
    while training_task.returncode is None:
        # 获取分布式和单机训练的最新指标
        dist_metrics = await get_metrics_distributed()
        single_metrics = await get_metrics_single()

        # 向所有连接的客户端推送数据
        for client in clients:
            try:
                await client.send_text(json.dumps({
                    "type": "metrics",
                    "distributed": dist_metrics,
                    "single": single_metrics
                }))
            except Exception:
                # 发送失败可能是客户端断开连接
                pass

        # 每秒更新一次
        await asyncio.sleep(1.0)


# ==================== 视频生成 API ====================


def _create_video_script() -> Path:
    """创建临时视频生成脚本"""
    script_content = '''#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from drl.video_generator import generate_video

print("[Video Script] Starting...", flush=True)

# 生成分布式视频
config_path_distributed = str(Path(__file__).parent / "config" / "config.yaml")
output_path_distributed = str(Path(__file__).parent / "output" / "video_distributed.mp4")
print(f"[Video Script] Generating distributed video...", flush=True)
result_dist = generate_video(
    config_path=config_path_distributed,
    output_path=output_path_distributed,
    num_episodes=1,
    max_steps=50,
    fps=30
)
print(f"[Video Script] Distributed result: {result_dist}", flush=True)

# 生成单机视频
config_path_single = str(Path(__file__).parent / "config" / "config_single.yaml")
output_path_single = str(Path(__file__).parent / "output" / "video_single.mp4")
print(f"[Video Script] Generating single video...", flush=True)
result_single = generate_video(
    config_path=config_path_single,
    output_path=output_path_single,
    num_episodes=1,
    max_steps=50,
    fps=30
)
print(f"[Video Script] Single result: {result_single}", flush=True)

print("[Video Script] Done!", flush=True)
'''
    script_path = REPO_ROOT / "temp_generate_videos.py"
    script_path.write_text(script_content)
    return script_path


async def _monitor_video_process(process: asyncio.subprocess.Process) -> None:
    """监控视频生成进程"""
    global video_generation_status
    
    try:
        video_generation_status = {"status": "generating", "progress": 0}
        
        # 读取输出
        while process.returncode is None:
            await asyncio.sleep(0.5)
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            video_generation_status = {
                "status": "completed",
                "progress": 100
            }
            print("[Server] Video generation completed successfully!", flush=True)
        else:
            error_msg = stderr.decode() if stderr else "Unknown error"
            video_generation_status = {
                "status": "error",
                "error": error_msg
            }
            print(f"[Server] Video generation failed: {error_msg}", flush=True)
            
    except Exception as e:
        print(f"[Server] Error monitoring video process: {e}", flush=True)
        import traceback
        traceback.print_exc()
        video_generation_status = {
            "status": "error",
            "error": str(e)
        }


@app.post("/api/videos/generate")
async def generate_videos() -> dict[str, Any]:
    """一键生成对比视频"""
    global video_generation_process, video_generation_status
    print("[Server] Received generate videos request")
    
    # 如果已有进程在运行，先终止它
    if video_generation_process and video_generation_process.returncode is None:
        print("[Server] Terminating existing video generation process...", flush=True)
        video_generation_process.terminate()
        try:
            await asyncio.wait_for(video_generation_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            video_generation_process.kill()
            await video_generation_process.wait()
    
    # 创建临时脚本
    script_path = _create_video_script()
    
    # 重置状态
    video_generation_status = {"status": "idle"}
    
    print(f"[Server] Starting video generation process...", flush=True)
    
    # 使用子进程运行视频生成
    video_generation_process = await asyncio.create_subprocess_exec(
        sys.executable, str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(REPO_ROOT)
    )
    
    # 启动监控任务
    asyncio.create_task(_monitor_video_process(video_generation_process))
    
    print(f"[Server] Video generation process started with PID: {video_generation_process.pid}", flush=True)
    
    return {"status": "started"}


@app.get("/api/videos/status")
async def get_video_status() -> dict[str, Any]:
    """获取视频生成状态"""
    return video_generation_status


@app.get("/api/videos/distributed", response_model=None)
async def get_distributed_video() -> Any:
    """获取分布式训练视频"""
    video_path = OUTPUT_DIR / "video_distributed.mp4"
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": "Video not found"}


@app.get("/api/videos/single", response_model=None)
async def get_single_video() -> Any:
    """获取单机训练视频"""
    video_path = OUTPUT_DIR / "video_single.mp4"
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": "Video not found"}


if __name__ == "__main__":
    import uvicorn
    # 启动 Web 服务器
    uvicorn.run(app, host="127.0.0.1", port=8000)