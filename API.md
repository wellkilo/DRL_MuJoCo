# DRL MuJoCo Web API

FastAPI 默认监听 `0.0.0.0:8000`。接口当前无鉴权，设计用于本机或 SSH 隧道访问，不应直接暴露到公网。

基础地址：

```text
http://localhost:8000
```

除视频文件外，响应均为 JSON。启动类接口当前以 HTTP 200 返回业务状态，调用方必须检查 `status` 字段。

## 环境与指标

### `GET /api/environments`

```bash
curl 'http://localhost:8000/api/environments'
```

```json
{
  "environments": {
    "hopper": {
      "name": "Hopper-v5",
      "description": "...",
      "difficulty": "..."
    }
  },
  "active": "hopper"
}
```

### `GET /api/metrics/distributed?env=<env>`

### `GET /api/metrics/single?env=<env>`

`env` 可取 `hopper`、`walker2d`、`halfcheetah`，缺省为 `hopper`。返回 CSV 行转换后的 JSON 数组。

```bash
curl 'http://localhost:8000/api/metrics/distributed?env=hopper'
```

```json
[
  {
    "step": 0,
    "elapsed_sec": 2.1,
    "total_steps": 16384.0,
    "sps": 7801.9,
    "episodes": 12,
    "avg_return": 18.2,
    "buffer_size": 16384.0,
    "loss": 0.31,
    "num_gpus": 1.0
  }
]
```

## 训练控制

### `POST /api/training/distributed/start?env=<env>`

### `POST /api/training/single/start?env=<env>`

请求无 JSON body：

```bash
curl -X POST 'http://localhost:8000/api/training/distributed/start?env=hopper'
```

成功：

```json
{"status": "started"}
```

重复启动：

```json
{"status": "already running"}
```

立即崩溃：

```json
{
  "status": "error",
  "returncode": 1,
  "error_detail": "Traceback ..."
}
```

### `POST /api/training/stop?env=<env>`

省略 `env` 会停止当前服务进程管理的全部训练。

```bash
curl -X POST 'http://localhost:8000/api/training/stop?env=hopper'
```

```json
{"status": "stopped", "environments": ["hopper"]}
```

### `GET /api/training/status?env=<env>`

```json
{"env": "hopper", "running": true}
```

不传 `env`：

```json
{"statuses": {"hopper": true, "walker2d": false, "halfcheetah": false}}
```

### `GET /api/training/logs?env=<env>&lines=<n>`

返回内存中的最近训练输出。`lines` 缺省为 50。

```json
{"env": "hopper", "lines": ["[Main] ..."], "total": 120}
```

## GPU 扩展实验

### `GET /api/scaling/configs`

扫描 `config/scaling/*.yaml`：

```json
{
  "configs": {
    "hopper_gpu4": {
      "env": "hopper",
      "num_gpus": 4,
      "config_path": "config/scaling/hopper_gpu4.yaml",
      "metrics_path": "output/scaling/hopper_gpu4/metrics.csv"
    }
  }
}
```

### `GET /api/scaling/metrics?config_name=<name>`

### `POST /api/scaling/start?config_name=<name>`

```bash
curl -X POST 'http://localhost:8000/api/scaling/start?config_name=hopper_gpu4'
```

```json
{
  "status": "started",
  "config": {"env": "hopper", "num_gpus": 4, "config_path": "..."}
}
```

### `GET /api/scaling/status`

```json
{"statuses": {"hopper_gpu4": true}}
```

### `GET /api/cluster/info`

```json
{
  "gpu_count": 1,
  "gpus": [{"index": 0, "name": "NVIDIA ...", "memory_gb": 24.0}],
  "ray_resources": {}
}
```

## 视频

### `POST /api/videos/generate?env=<env>`

异步生成单机与分布式策略视频。

### `GET /api/videos/status?env=<env>`

返回 `idle`、`generating`、`completed` 或 `error` 状态及错误信息。

### `GET /api/videos/distributed?env=<env>`

### `GET /api/videos/single?env=<env>`

成功时返回 MP4 文件；文件不存在时返回 JSON：

```json
{"error": "Video not found"}
```

## WebSocket

### `WS /ws/training`

连接：

```text
ws://localhost:8000/ws/training
```

服务端每秒推送当前活跃环境的指标：

```json
{
  "type": "metrics",
  "env": "hopper",
  "distributed": [],
  "single": []
}
```

训练退出时推送：

```json
{
  "type": "training_stopped",
  "env": "hopper",
  "returncode": 1,
  "error_detail": "可选错误详情"
}
```
