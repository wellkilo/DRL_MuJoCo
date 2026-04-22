# DRL MuJoCo Web UI 服务器
#
# 提供 Web 界面实时监控分布式训练过程
# 支持多环境：Hopper-v5, Walker2d-v5, HalfCheetah-v5
# 功能：启动/停止训练、实时可视化训练指标、对比单机与分布式性能、视频演示
#
# 前端使用 Next.js + Tailwind CSS 构建，通过 `next build` 生成静态导出到 web/out/

from __future__ import annotations

import asyncio
import csv
import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from drl.video_generator import generate_comparison_videos
except ImportError:
    generate_comparison_videos = None
    print("[Server] Warning: drl.video_generator not available, video generation disabled", flush=True)

# 初始化 FastAPI 应用
app = FastAPI(title="DRL MuJoCo Web UI")

# 项目路径配置
REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Next.js 静态导出目录
WEB_DIST = REPO_ROOT / "web" / "out"

# 环境配置注册表
ENVIRONMENTS = {
    "hopper": {
        "name": "Hopper-v5",
        "description": "单腿跳跃机器人",
        "difficulty": 1,
        "distributed_config": "config/config.yaml",
        "single_config": "config/config_single.yaml",
        "metrics_dir": "output",
        "dist_metrics": "metrics.csv",
        "single_metrics": "metrics_single.csv",
        "dist_video": "video_distributed.mp4",
        "single_video": "video_single.mp4",
    },
    "walker2d": {
        "name": "Walker2d-v5",
        "description": "双腿行走机器人",
        "difficulty": 2,
        "distributed_config": "config/walker2d.yaml",
        "single_config": "config/walker2d_single.yaml",
        "metrics_dir": "output/walker2d",
        "dist_metrics": "metrics.csv",
        "single_metrics": "metrics_single.csv",
        "dist_video": "video_distributed.mp4",
        "single_video": "video_single.mp4",
    },
    "halfcheetah": {
        "name": "HalfCheetah-v5",
        "description": "半猎豹奔跑机器人",
        "difficulty": 3,
        "distributed_config": "config/halfcheetah.yaml",
        "single_config": "config/halfcheetah_single.yaml",
        "metrics_dir": "output/halfcheetah",
        "dist_metrics": "metrics.csv",
        "single_metrics": "metrics_single.csv",
        "dist_video": "video_distributed.mp4",
        "single_video": "video_single.mp4",
    },
}

# 支持 GPU 信息检测
import torch
AVAILABLE_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"[Server] Available GPUs: {AVAILABLE_GPUS}", flush=True)

# 全局变量
training_tasks: dict[str, asyncio.subprocess.Process] = {}  # 每个环境的训练进程
active_env: str = "hopper"  # 当前活跃环境
clients: list[WebSocket] = []  # WebSocket 客户端列表
video_generation_processes: dict[str, asyncio.subprocess.Process] = {}  # 每个环境的视频生成进程
video_generation_status: dict[str, dict[str, Any]] = {}  # 每个环境的视频生成状态
training_output: dict[str, list[str]] = {}  # 每个环境的最近训练输出（用于错误诊断）


def get_env_config(env: str) -> dict:
    """获取环境配置，如果不存在则返回默认 hopper"""
    return ENVIRONMENTS.get(env, ENVIRONMENTS["hopper"])


def get_metrics_path(env: str, mode: str) -> Path:
    """获取指标文件路径"""
    cfg = get_env_config(env)
    metrics_dir = REPO_ROOT / cfg["metrics_dir"]
    filename = cfg["dist_metrics"] if mode == "distributed" else cfg["single_metrics"]
    return metrics_dir / filename


def load_metrics_file(metrics_path: Path) -> list[dict[str, Any]]:
    """从 CSV 文件加载训练指标"""
    if not metrics_path.exists():
        return []
    metrics: list[dict[str, Any]] = []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted_row = {}
            for k, v in row.items():
                if v == "":
                    converted_row[k] = float("nan")
                elif k in ("step", "episodes"):
                    converted_row[k] = int(float(v))
                else:
                    converted_row[k] = float(v)
            metrics.append(converted_row)
    return metrics


# ==================== API 端点 ====================

@app.get("/")
async def get_index() -> FileResponse:
    """返回主页"""
    if (WEB_DIST / "index.html").exists():
        return FileResponse(WEB_DIST / "index.html")
    return FileResponse(REPO_ROOT / "web" / "index.html")


@app.get("/api/environments")
async def get_environments() -> dict[str, Any]:
    """获取所有可用环境列表"""
    envs = {}
    for key, cfg in ENVIRONMENTS.items():
        envs[key] = {
            "name": cfg["name"],
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
        }
    return {"environments": envs, "active": active_env}


@app.get("/api/metrics/distributed")
async def get_metrics_distributed(env: str = Query("hopper")) -> list[dict[str, Any]]:
    """获取分布式训练指标"""
    return load_metrics_file(get_metrics_path(env, "distributed"))


@app.get("/api/metrics/single")
async def get_metrics_single(env: str = Query("hopper")) -> list[dict[str, Any]]:
    """获取单机训练指标"""
    return load_metrics_file(get_metrics_path(env, "single"))


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket) -> None:
    """WebSocket 端点：实时推送训练指标"""
    await websocket.accept()
    clients.append(websocket)
    print(f"[WebSocket] New client connected. Total clients: {len(clients)}", flush=True)
    try:
        while True:
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"[WebSocket] Error: {e}", flush=True)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        print(f"[WebSocket] Client disconnected. Total clients: {len(clients)}", flush=True)


@app.post("/api/training/distributed/start")
async def start_training_distributed(env: str = Query("hopper")) -> dict[str, str]:
    """启动分布式训练"""
    global active_env
    cfg = get_env_config(env)

    if env in training_tasks and training_tasks[env].returncode is None:
        return {"status": "already running"}

    # 确保输出目录存在
    metrics_dir = REPO_ROOT / cfg["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    active_env = env
    config_path = REPO_ROOT / cfg["distributed_config"]

    training_output[env] = []  # 初始化输出缓冲区
    training_tasks[env] = await asyncio.create_subprocess_exec(
        sys.executable, str(REPO_ROOT / "main.py"), str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    asyncio.create_task(monitor_training(env))
    asyncio.create_task(monitor_training_output(env))
    return {"status": "started"}


@app.post("/api/training/single/start")
async def start_training_single(env: str = Query("hopper")) -> dict[str, str]:
    """启动单机训练"""
    global active_env
    cfg = get_env_config(env)

    if env in training_tasks and training_tasks[env].returncode is None:
        return {"status": "already running"}

    metrics_dir = REPO_ROOT / cfg["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    active_env = env
    config_path = REPO_ROOT / cfg["single_config"]

    training_output[env] = []  # 初始化输出缓冲区
    training_tasks[env] = await asyncio.create_subprocess_exec(
        sys.executable, str(REPO_ROOT / "main.py"), str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    asyncio.create_task(monitor_training(env))
    asyncio.create_task(monitor_training_output(env))
    return {"status": "started"}


@app.post("/api/training/stop")
async def stop_training(env: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """停止训练进程"""
    # If no env specified, stop all running tasks
    envs_to_stop = [env] if env else list(training_tasks.keys())
    stopped = []

    for e in envs_to_stop:
        task = training_tasks.get(e)
        if task is None:
            continue
        try:
            if task.returncode is not None:
                # Process already exited
                continue
            print(f"[Server] Stopping training for {e}...", flush=True)
            task.terminate()
            try:
                await asyncio.wait_for(task.wait(), timeout=15.0)
                print(f"[Server] Training for {e} exited gracefully", flush=True)
            except asyncio.TimeoutError:
                print(f"[Server] Timeout, forcing {e} to stop...", flush=True)
                task.kill()
                await task.wait()
            stopped.append(e)
        except Exception as exc:
            print(f"[Server] Error stopping {e}: {exc}", flush=True)
            stopped.append(e)

    return {"status": "stopped", "environments": stopped}


@app.get("/api/training/status")
async def get_training_status(env: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """查询训练进程状态，前端可按环境查询后端真实的训练进程状态"""
    if env:
        task = training_tasks.get(env)
        return {"env": env, "running": task is not None and task.returncode is None}
    statuses = {}
    for e in ENVIRONMENTS:
        task = training_tasks.get(e)
        statuses[e] = task is not None and task.returncode is None
    return {"statuses": statuses}


@app.get("/api/training/logs")
async def get_training_logs(env: str = Query("hopper"), lines: int = Query(50)) -> Dict[str, Any]:
    """获取训练进程的最近日志输出，用于错误诊断"""
    output_lines = training_output.get(env, [])
    # 返回最后 N 行
    recent = output_lines[-lines:] if output_lines else []
    return {"env": env, "lines": recent, "total": len(output_lines)}


# ==================== GPU 扩展实验 API ====================

def _discover_scaling_configs() -> dict:
    """扫描 config/scaling/ 目录, 发现可用的 GPU 扩展配置"""
    scaling_dir = REPO_ROOT / "config" / "scaling"
    configs: dict[str, Any] = {}
    if not scaling_dir.exists():
        return configs
    for yaml_file in sorted(scaling_dir.glob("*.yaml")):
        # 文件名格式: hopper_gpu4.yaml
        name = yaml_file.stem  # hopper_gpu4
        parts = name.rsplit("_gpu", 1)
        if len(parts) == 2:
            env_key = parts[0]
            try:
                num_gpus = int(parts[1])
            except ValueError:
                continue
            configs[name] = {
                "env": env_key,
                "num_gpus": num_gpus,
                "config_path": str(yaml_file.relative_to(REPO_ROOT)),
                "metrics_path": f"output/scaling/{name}/metrics.csv",
            }
    return configs


@app.get("/api/scaling/configs")
async def get_scaling_configs() -> dict[str, Any]:
    """获取所有可用的 GPU 扩展实验配置"""
    configs = _discover_scaling_configs()
    return {"configs": configs}


@app.get("/api/scaling/metrics")
async def get_scaling_metrics(config_name: str = Query(...)) -> list[dict[str, Any]]:
    """获取指定 GPU 扩展实验的 metrics"""
    configs = _discover_scaling_configs()
    if config_name not in configs:
        return []
    metrics_path = REPO_ROOT / configs[config_name]["metrics_path"]
    return load_metrics_file(metrics_path)


@app.post("/api/scaling/start")
async def start_scaling_experiment(config_name: str = Query(...)) -> dict[str, Any]:
    """启动 GPU 扩展实验"""
    configs = _discover_scaling_configs()
    if config_name not in configs:
        return {"status": "error", "message": f"Config {config_name} not found"}

    task_key = f"scaling_{config_name}"
    if task_key in training_tasks and training_tasks[task_key].returncode is None:
        return {"status": "already running"}

    cfg = configs[config_name]
    config_path = REPO_ROOT / cfg["config_path"]

    # 确保输出目录
    metrics_dir = REPO_ROOT / "output" / "scaling" / config_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    training_output[task_key] = []
    training_tasks[task_key] = await asyncio.create_subprocess_exec(
        sys.executable, str(REPO_ROOT / "main.py"), str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    asyncio.create_task(monitor_training_output(task_key))
    return {"status": "started", "config": cfg}


@app.get("/api/scaling/status")
async def get_scaling_status() -> dict[str, Any]:
    """获取所有 GPU 扩展实验的运行状态"""
    statuses: dict[str, bool] = {}
    for key, task in training_tasks.items():
        if key.startswith("scaling_"):
            config_name = key.replace("scaling_", "")
            statuses[config_name] = task.returncode is None
    return {"statuses": statuses}


@app.get("/api/cluster/info")
async def get_cluster_info() -> dict[str, Any]:
    """获取集群 GPU 信息 (用于 UI 展示)"""
    gpu_info: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_mem / 1e9, 1),
            })

    ray_resources: dict[str, Any] = {}
    try:
        import ray as _ray
        if _ray.is_initialized():
            ray_resources = dict(_ray.cluster_resources())
    except Exception:
        pass

    return {
        "gpu_count": len(gpu_info),
        "gpus": gpu_info,
        "ray_resources": ray_resources,
    }


# ==================== 视频生成 API ====================

def _create_video_script(env: str) -> Path:
    """创建临时视频生成脚本"""
    cfg = get_env_config(env)
    script_content = f'''#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from drl.video_generator import generate_video

print("[Video Script] Starting for {env}...", flush=True)
TARGET_DURATION = 15.0

# 生成分布式视频
config_path_distributed = str(Path(__file__).parent / "{cfg['distributed_config']}")
output_dir = Path(__file__).parent / "{cfg['metrics_dir']}"
output_dir.mkdir(parents=True, exist_ok=True)
output_path_distributed = str(output_dir / "{cfg['dist_video']}")
print(f"[Video Script] Generating distributed video for {env}...", flush=True)
generate_video(config_path=config_path_distributed, output_path=output_path_distributed, target_duration=TARGET_DURATION, fps=30)

# 生成单机视频
config_path_single = str(Path(__file__).parent / "{cfg['single_config']}")
output_path_single = str(output_dir / "{cfg['single_video']}")
print(f"[Video Script] Generating single video for {env}...", flush=True)
generate_video(config_path=config_path_single, output_path=output_path_single, target_duration=TARGET_DURATION, fps=30)

print("[Video Script] Done for {env}!", flush=True)
'''
    script_path = REPO_ROOT / f"temp_generate_videos_{env}.py"
    script_path.write_text(script_content)
    return script_path


async def _monitor_video_process(env: str, process: asyncio.subprocess.Process) -> None:
    """监控视频生成进程"""
    try:
        video_generation_status[env] = {"status": "generating", "progress": 0}
        _, stderr = await process.communicate()
        if process.returncode == 0:
            video_generation_status[env] = {"status": "completed", "progress": 100}
            print(f"[Server] Video generation for {env} completed!", flush=True)
        else:
            error_msg = (stderr or b"").decode(errors="replace").strip()
            if not error_msg:
                error_msg = f"Process exited with code {process.returncode}"
            video_generation_status[env] = {"status": "error", "error": error_msg}
            print(f"[Server] Video generation for {env} failed: {error_msg}", flush=True)
    except BrokenPipeError:
        # BrokenPipeError is harmless — process output pipe closed before we read it
        if process.returncode == 0:
            video_generation_status[env] = {"status": "completed", "progress": 100}
        else:
            video_generation_status[env] = {"status": "error", "error": "BrokenPipeError"}
    except Exception as e:
        video_generation_status[env] = {"status": "error", "error": str(e)}


@app.post("/api/videos/generate")
async def generate_videos(env: str = Query("hopper")) -> dict[str, Any]:
    """一键生成对比视频"""
    cfg = get_env_config(env)

    # 终止已有进程
    if env in video_generation_processes and video_generation_processes[env].returncode is None:
        video_generation_processes[env].terminate()
        try:
            await asyncio.wait_for(video_generation_processes[env].wait(), timeout=5.0)
        except asyncio.TimeoutError:
            video_generation_processes[env].kill()
            await video_generation_processes[env].wait()

    script_path = _create_video_script(env)
    video_generation_status[env] = {"status": "idle"}

    video_generation_processes[env] = await asyncio.create_subprocess_exec(
        sys.executable, str(script_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(REPO_ROOT)
    )

    asyncio.create_task(_monitor_video_process(env, video_generation_processes[env]))
    return {"status": "started"}


@app.get("/api/videos/status")
async def get_video_status(env: str = Query("hopper")) -> dict[str, Any]:
    """获取视频生成状态"""
    return video_generation_status.get(env, {"status": "idle"})


@app.get("/api/videos/distributed", response_model=None)
async def get_distributed_video(env: str = Query("hopper")) -> Any:
    """获取分布式训练视频"""
    cfg = get_env_config(env)
    video_path = REPO_ROOT / cfg["metrics_dir"] / cfg["dist_video"]
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": "Video not found"}


@app.get("/api/videos/single", response_model=None)
async def get_single_video(env: str = Query("hopper")) -> Any:
    """获取单机训练视频"""
    cfg = get_env_config(env)
    video_path = REPO_ROOT / cfg["metrics_dir"] / cfg["single_video"]
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": "Video not found"}


# ==================== 监控任务 ====================

async def monitor_training_output(env: str) -> None:
    """监控训练进程的输出，并保存最近输出用于错误诊断"""
    task = training_tasks.get(env)
    if not task or not task.stdout:
        return
    print(f"[Training Output] Monitoring output for {env}...", flush=True)
    try:
        while True:
            line = await task.stdout.readline()
            if not line:
                break
            decoded_line = line.decode().rstrip()
            print(f"[{env}] {decoded_line}", flush=True)
            # 保存最近的输出行（保留最后200行用于错误诊断）
            if env not in training_output:
                training_output[env] = []
            training_output[env].append(decoded_line)
            if len(training_output[env]) > 200:
                training_output[env] = training_output[env][-200:]
    except Exception as e:
        print(f"[Training Output] Error for {env}: {e}", flush=True)


def _extract_error_lines(output_lines: list[str], max_lines: int = 30) -> str:
    """从训练输出中提取错误信息，优先提取 Traceback 及其后续行"""
    if not output_lines:
        return ""
    
    # 查找最后一个 Traceback 的位置
    traceback_idx = -1
    for i in range(len(output_lines) - 1, -1, -1):
        if "Traceback" in output_lines[i] or "Error:" in output_lines[i] or "Exception:" in output_lines[i]:
            traceback_idx = i
            break
    
    if traceback_idx >= 0:
        # 从 Traceback 开始取到末尾（或最多 max_lines 行）
        error_lines = output_lines[traceback_idx:traceback_idx + max_lines]
    else:
        # 没有 Traceback，取最后 max_lines 行
        error_lines = output_lines[-max_lines:]
    
    return "\n".join(error_lines)


async def monitor_training(env: str) -> None:
    """监控训练进程并实时推送指标"""
    task = training_tasks.get(env)
    if not task:
        return

    print(f"[Monitor] Starting monitoring for {env}...", flush=True)
    while True:
        # Check if process is still running
        if task.returncode is not None:
            print(f"[Monitor] Training for {env} ended with returncode {task.returncode}.", flush=True)
            
            # 提取错误信息
            error_detail = ""
            if task.returncode != 0:
                output_lines = training_output.get(env, [])
                error_detail = _extract_error_lines(output_lines)
                if error_detail:
                    print(f"[Monitor] Error detail for {env}:\n{error_detail}", flush=True)
            
            # Notify all clients that training stopped
            for client in clients.copy():
                try:
                    msg = {
                        "type": "training_stopped",
                        "env": env,
                        "returncode": task.returncode,
                    }
                    if error_detail:
                        msg["error_detail"] = error_detail
                    await client.send_text(json.dumps(msg))
                except Exception:
                    if client in clients:
                        clients.remove(client)
            break

        dist_path = get_metrics_path(env, "distributed")
        single_path = get_metrics_path(env, "single")
        dist_metrics = load_metrics_file(dist_path)
        single_metrics = load_metrics_file(single_path)

        for client in clients.copy():
            try:
                await client.send_text(json.dumps({
                    "type": "metrics",
                    "env": env,
                    "distributed": dist_metrics,
                    "single": single_metrics,
                }))
            except Exception:
                if client in clients:
                    clients.remove(client)

        await asyncio.sleep(1.0)


# ==================== 静态资源配置 ====================

app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

if WEB_DIST.exists():
    next_static = WEB_DIST / "_next"
    if next_static.exists():
        app.mount("/_next", StaticFiles(directory=next_static), name="next_static")

    @app.get("/icon.svg")
    async def get_icon_svg() -> FileResponse:
        icon_path = WEB_DIST / "icon.svg"
        if icon_path.exists():
            return FileResponse(icon_path, media_type="image/svg+xml")
        return FileResponse(REPO_ROOT / "web" / "src" / "app" / "icon.svg", media_type="image/svg+xml")


if __name__ == "__main__":
    import os
    import uvicorn
    host = os.environ.get("WEBUI_HOST", "0.0.0.0")    # Slurm 需要 0.0.0.0
    port = int(os.environ.get("WEBUI_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
