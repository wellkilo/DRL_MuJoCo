from __future__ import annotations

import asyncio
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="DRL MuJoCo Web UI")

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=REPO_ROOT / "web/static"), name="static")
templates = Jinja2Templates(directory=REPO_ROOT / "web/templates")

training_task: asyncio.subprocess.Process | None = None
clients: list[WebSocket] = []


@app.get("/")
async def get_index() -> FileResponse:
    return FileResponse(REPO_ROOT / "web/templates/index.html")


@app.get("/api/metrics")
async def get_metrics() -> list[dict[str, Any]]:
    metrics_path = OUTPUT_DIR / "metrics.csv"
    if not metrics_path.exists():
        return []
    metrics: list[dict[str, Any]] = []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append({k: float(v) if k not in ("step", "episodes") else int(float(v)) for k, v in row.items()})
    return metrics


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket) -> None:
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)
    except Exception:
        pass
    finally:
        clients.remove(websocket)


@app.post("/api/training/start")
async def start_training() -> dict[str, str]:
    global training_task
    if training_task and training_task.returncode is None:
        return {"status": "already running"}
    training_task = await asyncio.create_subprocess_exec(
        "python", str(REPO_ROOT / "main.py"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    asyncio.create_task(monitor_training())
    return {"status": "started"}


@app.post("/api/training/stop")
async def stop_training() -> dict[str, str]:
    global training_task
    if training_task and training_task.returncode is None:
        training_task.terminate()
        try:
            await asyncio.wait_for(training_task.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            training_task.kill()
            await training_task.wait()
    return {"status": "stopped"}


async def monitor_training() -> None:
    if not training_task:
        return
    while training_task.returncode is None:
        metrics = await get_metrics()
        for client in clients:
            try:
                await client.send_text(json.dumps({"type": "metrics", "data": metrics}))
            except Exception:
                pass
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
