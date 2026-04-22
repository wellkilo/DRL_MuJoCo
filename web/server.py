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
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware  # [FIX] 新增：开启 CORS，允许 Next.js dev server 跨域访问
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

# [FIX] 开启 CORS，允许 Next.js dev server（:3000）直接跨域调用后端 API + WebSocket
#       不加这一段 dev 模式下所有 POST 请求都会被浏览器 preflight 拦截
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        with metrics_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                converted_row = {}
                for k, v in row.items():
                    if v == "":
                        converted_row[k] = float("nan")
                    elif k in ("step", "episodes"):
                        try:
                            converted_row[k] = int(float(v))
                        except ValueError:
                            converted_row[k] = 0
                    else:
                        try:
                            converted_row[k] = float(v)
                        except ValueError:
                            converted_row[k] = float("nan")
                metrics.append(converted_row)
    except Exception as e:
        # [FIX] CSV 读取加 try/except，避免训练正在写入时短暂读到半行导致 WebSocket 中断
        print(f"[Server] load_metrics_file error: {e}", flush=True)
        return metrics
    return metrics


# [FIX] 新增：统一的子进程启动辅助函数，确保 PYTHONUNBUFFERED + python -u + env 继承
def _build_child_env() -> dict:
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    return child_env


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
    """
    WebSocket 端点：实时推送训练指标。

    [FIX] 核心修复：握手后即启动常驻推送循环，不再依赖训练启动时才创建的 monitor_training。
          这样即使用户刷新页面、切换 tab、或训练尚未启动，前端仍能持续收到最新 CSV 数据，
          彻底解决"图表没有数据同步更新"。
    """
    await websocket.accept()
    clients.append(websocket)
    print(f"[WebSocket] New client connected. Total clients: {len(clients)}", flush=True)
    try:
        while True:
            # 推送当前 active_env 的最新指标
            try:
                dist = load_metrics_file(get_metrics_path(active_env, "distributed"))
                single = load_metrics_file(get_metrics_path(active_env, "single"))
                await websocket.send_text(json.dumps({
                    "type": "metrics",
                    "env": active_env,
                    "distributed": dist,
                    "single": single,
                }))
            except Exception as send_err:
                print(f"[WebSocket] Send error: {send_err}", flush=True)
                break

            # 用 wait_for(receive) 实现可中断等待：
            # - 超时到 → 继续推送
            # - 收到客户端消息 → 正常忽略（可扩展为心跳）
            # - 客户端断开 → 立刻退出循环
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
    except Exception as e:
        print(f"[WebSocket] Error: {e}", flush=True)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        print(f"[WebSocket] Client disconnected. Total clients: {len(clients)}", flush=True)


@app.post("/api/training/distributed/start")
async def start_training_distributed(env: str = Query("hopper")) -> dict[str, Any]:
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

    if not config_path.exists():
        return {"status": "error", "message": f"Config not found: {config_path}"}

    training_output[env] = []  # 初始化输出缓冲区

    # [FIX] 加 -u 强制无缓冲 + 传递 env（含 PYTHONUNBUFFERED=1），避免 stdout 被 block-buffered 吞掉错误
    training_tasks[env] = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(REPO_ROOT / "main.py"), str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=_build_child_env(),
    )

    asyncio.create_task(monitor_training(env))
    asyncio.create_task(monitor_training_output(env))

    # [FIX] 启动后等 1.2s，若训练进程立刻崩溃则把错误同步返回给前端
    #       这样用户点按钮后能立即看到"Config Error""ModuleNotFoundError"等信息，
    #       而不是"按钮没反应 → 过几秒自己恢复"的诡异现象
    await asyncio.sleep(1.2)
    proc = training_tasks[env]
    if proc.returncode is not None and proc.returncode != 0:
        err = _extract_error_lines(training_output.get(env, []))
        return {
            "status": "error",
            "returncode": proc.returncode,
            "error_detail": err or "Process exited immediately with no output",
        }

    return {"status": "started"}


@app.post("/api/training/single/start")
async def start_training_single(env: str = Query("hopper")) -> dict[str, Any]:
    """启动单机训练"""
    global active_env
    cfg = get_env_config(env)

    if env in training_tasks and training_tasks[env].returncode is None:
        return {"status": "already running"}

    metrics_dir = REPO_ROOT / cfg["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    active_env = env
    config_path = REPO_ROOT / cfg["single_config"]

    if not config_path.exists():
        return {"status": "error", "message": f"Config not found: {config_path}"}

    training_output[env] = []

    training_tasks[env] = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(REPO_ROOT / "main.py"), str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=_build_child_env(),
    )

    asyncio.create_task(monitor_training(env))
    asyncio.create_task(monitor_training_output(env))

    await asyncio.sleep(1.2)
    proc = training_tasks[env]
    if proc.returncode is not None and proc.returncode != 0:
        err = _extract_error_lines(training_output.get(env, []))
        return {
            "status": "error",
            "returncode": proc.returncode,
            "error_detail": err or "Process exited immediately with no output",
        }

    return {"status": "started"}


@app.post("/api/training/stop")
async def stop_training(env: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """停止训练进程"""
    envs_to_stop = [env] if env else list(training_tasks.keys())
    stopped = []

    for e in envs_to_stop:
        task = training_tasks.get(e)
        if task is None:
            continue
        try:
            if task.returncode is not None:
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
    """查询训练进程状态"""
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
        name = yaml_file.stem
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
    configs = _discover_scaling_configs()
    return {"configs": configs}


@app.get("/api/scaling/metrics")
async def get_scaling_metrics(config_name: str = Query(...)) -> list[dict[str, Any]]:
    configs = _discover_scaling_configs()
    if config_name not in configs:
        return []
    metrics_path = REPO_ROOT / configs[config_name]["metrics_path"]
    return load_metrics_file(metrics_path)


@app.post("/api/scaling/start")
async def start_scaling_experiment(config_name: str = Query(...)) -> dict[str, Any]:
    configs = _discover_scaling_configs()
    if config_name not in configs:
        return {"status": "error", "message": f"Config {config_name} not found"}

    task_key = f"scaling_{config_name}"
    if task_key in training_tasks and training_tasks[task_key].returncode is None:
        return {"status": "already running"}

    cfg = configs[config_name]
    config_path = REPO_ROOT / cfg["config_path"]

    metrics_dir = REPO_ROOT / "output" / "scaling" / config_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    training_output[task_key] = []
    training_tasks[task_key] = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(REPO_ROOT / "main.py"), str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=_build_child_env(),
    )

    asyncio.create_task(monitor_training_output(task_key))

    await asyncio.sleep(1.2)
    proc = training_tasks[task_key]
    if proc.returncode is not None and proc.returncode != 0:
        err = _extract_error_lines(training_output.get(task_key, []))
        return {"status": "error", "returncode": proc.returncode, "error_detail": err}

    return {"status": "started", "config": cfg}


@app.get("/api/scaling/status")
async def get_scaling_status() -> dict[str, Any]:
    statuses: dict[str, bool] = {}
    for key, task in training_tasks.items():
        if key.startswith("scaling_"):
            config_name = key.replace("scaling_", "")
            statuses[config_name] = task.returncode is None
    return {"statuses": statuses}


@app.get("/api/cluster/info")
async def get_cluster_info() -> dict[str, Any]:
    gpu_info: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / 1e9, 1),  # [FIX] total_mem -> total_memory（PyTorch 正确字段名）
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
    cfg = get_env_config(env)
    script_content = f'''#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from drl.video_generator import generate_video

print("[Video Script] Starting for {env}...", flush=True)
TARGET_DURATION = 15.0

config_path_distributed = str(Path(__file__).parent / "{cfg['distributed_config']}")
output_dir = Path(__file__).parent / "{cfg['metrics_dir']}"
output_dir.mkdir(parents=True, exist_ok=True)
output_path_distributed = str(output_dir / "{cfg['dist_video']}")
print(f"[Video Script] Generating distributed video for {env}...", flush=True)
generate_video(config_path=config_path_distributed, output_path=output_path_distributed, target_duration=TARGET_DURATION, fps=30)

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
        if process.returncode == 0:
            video_generation_status[env] = {"status": "completed", "progress": 100}
        else:
            video_generation_status[env] = {"status": "error", "error": "BrokenPipeError"}
    except Exception as e:
        video_generation_status[env] = {"status": "error", "error": str(e)}


@app.post("/api/videos/generate")
async def generate_videos(env: str = Query("hopper")) -> dict[str, Any]:
    cfg = get_env_config(env)

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
        sys.executable, "-u", str(script_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(REPO_ROOT),
        env=_build_child_env(),
    )

    asyncio.create_task(_monitor_video_process(env, video_generation_processes[env]))
    return {"status": "started"}


@app.get("/api/videos/status")
async def get_video_status(env: str = Query("hopper")) -> dict[str, Any]:
    return video_generation_status.get(env, {"status": "idle"})


@app.get("/api/videos/distributed", response_model=None)
async def get_distributed_video(env: str = Query("hopper")) -> Any:
    cfg = get_env_config(env)
    video_path = REPO_ROOT / cfg["metrics_dir"] / cfg["dist_video"]
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": "Video not found"}


@app.get("/api/videos/single", response_model=None)
async def get_single_video(env: str = Query("hopper")) -> Any:
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
            decoded_line = line.decode(errors="replace").rstrip()
            print(f"[{env}] {decoded_line}", flush=True)
            if env not in training_output:
                training_output[env] = []
            training_output[env].append(decoded_line)
            if len(training_output[env]) > 200:
                training_output[env] = training_output[env][-200:]
    except Exception as e:
        print(f"[Training Output] Error for {env}: {e}", flush=True)


def _extract_error_lines(output_lines: list[str], max_lines: int = 30) -> str:
    if not output_lines:
        return ""
    traceback_idx = -1
    for i in range(len(output_lines) - 1, -1, -1):
        if "Traceback" in output_lines[i] or "Error:" in output_lines[i] or "Exception:" in output_lines[i]:
            traceback_idx = i
            break
    if traceback_idx >= 0:
        error_lines = output_lines[traceback_idx:traceback_idx + max_lines]
    else:
        error_lines = output_lines[-max_lines:]
    return "\n".join(error_lines)


async def monitor_training(env: str) -> None:
    """
    监控训练进程生命周期（仅负责：进程结束时通知前端 + 输出错误）

    [FIX] 该函数不再承担实时推送 metrics 的职责（推送职责已挪到 /ws/training 内），
          避免"训练结束 → monitor 退出 → 再无推送"的问题。
    """
    task = training_tasks.get(env)
    if not task:
        return

    print(f"[Monitor] Starting lifecycle monitor for {env}...", flush=True)
    while True:
        if task.returncode is not None:
            print(f"[Monitor] Training for {env} ended with returncode {task.returncode}.", flush=True)

            error_detail = ""
            if task.returncode != 0:
                output_lines = training_output.get(env, [])
                error_detail = _extract_error_lines(output_lines)
                if error_detail:
                    print(f"[Monitor] Error detail for {env}:\n{error_detail}", flush=True)

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
    import uvicorn
    # [FIX] 默认绑定 0.0.0.0，适配 Slurm 节点经 SSH port-forwarding 访问的场景
    host = os.environ.get("WEBUI_HOST", "0.0.0.0")
    port = int(os.environ.get("WEBUI_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
