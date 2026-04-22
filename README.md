<div align="center">

# 🤖 DRL MuJoCo

**分布式深度强化学习训练框架**

基于 Ray 的 Actor-Learner 架构，面向 MuJoCo 环境进行并行采样与策略优化

[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?logo=tailwindcss)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Ray](https://img.shields.io/badge/Ray-2.x-028CF0?logo=ray)](https://ray.io/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-000000?logo=rust)](https://rust-lang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

</div>

---

## ✨ 核心特性

<table>
<tr>
<td width="50%">

### 🚀 分布式训练
- **Actor-Learner 架构**：Ray 驱动的并行采样
- **Parameter Server**：异步参数同步
- **PPO 算法**：裁剪机制 + 自适应熵系数
- **GAE**：广义优势估计
- **自动设备选择**：CUDA → MPS → CPU

</td>
<td width="50%">

### 📊 实时可视化
- **Next.js 15** + **React 19** + **Tailwind CSS 3**
- **4 核心指标**：SPS、平均回报、损失、Buffer 大小
- **WebSocket 实时推送**：自动重连机制
- **性能对比**：分布式 vs 单机训练
- **一键生成视频**：智能体表现演示

</td>
</tr>
<tr>
<td width="50%">

### ⚡ Rust 高性能缓冲区
- **SoA 内存布局**：连续内存，缓存友好
- **Rayon 并行 GAE**：多线程优势函数计算
- **PyO3 绑定**：无缝集成 Python
- **目标 10-50x** 内存操作加速

</td>
<td width="50%">

### 🧠 PPO 优化
- 全局 Advantage 标准化
- 学习率线性衰减
- 梯度裁剪 + KL 散度早停
- 价值函数解释方差监控
- Clip Fraction 实时追踪

</td>
</tr>
<tr>
<td colspan="2">

### 🆕 多 GPU 扩展实验
- 🔥 **多 Learner 并行**：每 GPU 独占 1 Learner + 8 Actors，线性扩展吞吐
- 🔄 **参数平均同步**：ParameterServer 周期性平均所有 Learner 参数，保持策略一致
- 📐 **自动缩放配置**：`batch_size` / `buffer_capacity` 随 GPU 数等比扩展
- 🖥️ **Web UI 集成**：集群 GPU 信息展示 + 一键启动扩展实验 + SSH 端口转发
- 📦 **批量实验提交**：4 / 8 / 16 / 32 GPU 一键对比实验 + 结果汇总分析

</td>
</tr>
</table>

---

## 🖼️ 界面预览

<table>
<tr>
<td align="center"><b>📊 分布式训练监控</b></td>
<td align="center"><b>📈 单机训练监控</b></td>
<td align="center"><b>⚖️ 性能对比视图</b></td>
</tr>
<tr>
<td><img src="assets/web_ui_distributed.png" alt="Distributed" width="100%"></td>
<td><img src="assets/web_ui_single.png" alt="Single" width="100%"></td>
<td><img src="assets/web_ui_comparison.png" alt="Comparison" width="100%"></td>
</tr>
</table>

<div align="center">
<img src="assets/web_video_demo.gif" alt="Video Demo" width="60%">
<p><b>🎬 视频演示：一键生成分布式 & 单机训练结果对比视频</b></p>
</div>

---

## 🏗️ 项目结构

```
DRL_MuJoCo/
├── main.py                    # 训练入口（支持多 GPU 扩展）
├── drl/                       # 核心 DRL 模块
│   ├── config.py              # 配置 dataclass（含 GPU 扩展字段）
│   ├── config_loader.py       # YAML 配置加载
│   ├── models.py              # Actor-Critic 模型
│   ├── ray_components.py      # Ray 分布式组件（Learner 声明 num_gpus=1）
│   ├── logging_utils.py       # 日志工具
│   └── video_generator.py     # 视频生成
├── config/                    # 配置文件
│   ├── config.yaml            # 分布式训练 (8 Actors)
│   ├── config_single.yaml     # 单机训练 (1 Actor)
│   ├── walker2d.yaml          # Walker2d 分布式配置
│   ├── walker2d_single.yaml   # Walker2d 单机配置
│   ├── halfcheetah.yaml       # HalfCheetah 分布式配置
│   ├── halfcheetah_single.yaml # HalfCheetah 单机配置
│   └── scaling/               # 🆕 GPU 扩展实验配置（自动生成）
│       ├── hopper_gpu4.yaml
│       ├── hopper_gpu8.yaml
│       └── ...
├── web/                       # Web UI
│   ├── server.py              # FastAPI 后端（🆕 GPU 扩展 + 集群 API）
│   ├── next.config.ts         # Next.js 配置
│   ├── tailwind.config.ts     # Tailwind 主题配置
│   └── src/
│       ├── app/               # Next.js App Router
│       ├── components/        # React 组件
│       ├── hooks/             # 自定义 Hooks
│       ├── services/          # API & WebSocket
│       ├── stores/            # Zustand 状态
│       └── types/             # TypeScript 类型
├── rust_buffer/               # Rust 高性能缓冲区
│   ├── src/
│   │   ├── buffer.rs          # SoA 回放缓冲区
│   │   ├── gae.rs             # 并行 GAE 计算
│   │   └── lib.rs             # PyO3 绑定
│   ├── Cargo.toml
│   └── pyproject.toml
├── scripts/                   # 辅助脚本
│   ├── build.sh               # 环境搭建
│   ├── start.sh               # 交互式启动器
│   ├── plot_training.py       # 训练曲线
│   ├── plot_comparison.py     # 对比曲线
│   ├── gen_scaling_configs.py # 🆕 生成 GPU 扩展配置
│   ├── analyze_scaling.py     # 🆕 GPU 扩展结果分析
│   └── slurm/                 # Slurm 集群脚本
│       ├── setup_env.sh       #   Conda 环境创建 + 前端构建
│       ├── run_webui.sh       #   🆕 Web UI (Slurm 计算节点)
│       ├── run_single.sh      #   单机训练 sbatch
│       ├── run_distributed.sh #   分布式训练 sbatch
│       ├── run_all_envs.sh    #   批量提交 3环境×2模式
│       ├── run_scaling.sh     #   🆕 GPU 扩展实验 sbatch
│       ├── run_gpu_experiments.sh # 🆕 一键批量 GPU 对比实验
│       ├── run_plot.sh        #   训练后绘图
│       └── monitor.sh         #   作业监控工具
└── logs/                      # Slurm 日志目录
```

---

## 🚀 快速开始

### 1️⃣ 环境搭建

<details>
<summary>🐧 macOS / Linux</summary>

```bash
bash scripts/build.sh
```

</details>

<details>
<summary>🪟 Windows</summary>

> 只需确保已安装 [Miniforge](https://github.com/conda-forge/miniforge)（推荐）或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)，脚本会自动初始化 conda 环境

```cmd
scripts\build.bat
```

> **Windows 额外注意事项：**
> - MuJoCo 需要 [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)，一般 Windows 10/11 已预装
> - 如需构建 Rust Buffer，需安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（勾选 "C++ build tools"）和 [Rust](https://www.rust-lang.org/tools/install)
> - Ray 在 Windows 上支持单机多进程分布式训练（本项目模式），但不支持多节点集群

</details>

该脚本将自动：
- ✅ 检查 & 创建 conda 环境 (`drl-arm`)
- ✅ 安装 Python 依赖 (PyTorch, Ray, MuJoCo...)
- ✅ 可选构建 Next.js 前端
- ✅ 可选构建 Rust 回放缓冲区

### 2️⃣ 启动训练

<details>
<summary>🐧 macOS / Linux</summary>

```bash
bash scripts/start.sh
```

</details>

<details>
<summary>🪟 Windows</summary>

> 脚本会自动初始化 conda，直接在 CMD 中运行即可

```cmd
scripts\start.bat
```

</details>

| 选项 | 描述 |
|:----:|------|
| `1` | 分布式训练 (8 Actors) |
| `2` | 单机训练 (1 Actor) |
| `3` | 绘制训练曲线 |
| `4` | 绘制对比曲线 |
| `5` | 启动 Web UI (生产模式) |
| `6` | 🌟 启动 Next.js Dev Server + Web UI (推荐) |

### 3️⃣ 访问 Web UI

选择选项 `6` 后，打开浏览器访问 **http://localhost:3000**

---

## ⚙️ 配置说明

编辑 [`config/config.yaml`](config/config.yaml) 调整训练超参：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `env_name` | `Hopper-v5` | MuJoCo 环境名称 |
| `num_actors` | `8` | 并行采样器数量 |
| `rollout_length` | `2048` | 单次采样轨迹长度 |
| `batch_size` | `256` | 训练批次大小 |
| `lr` | `0.0003` | 学习率 |
| `gamma` | `0.99` | 折扣因子 |
| `gae_lambda` | `0.95` | GAE λ 参数 |
| `clip_ratio` | `0.15` | PPO 裁剪比率 |
| `hidden_sizes` | `[256, 256]` | 隐藏层大小 |
| `max_iters` | `1000` | 最大训练迭代数 |
| `lr_schedule` | `linear` | 学习率调度策略 |
| `target_kl` | `0.015` | KL 散度早停阈值 |
| `num_gpus` | `1` | 🆕 GPU 数量（每个 GPU 运行一个 Learner） |
| `actors_per_gpu` | `8` | 🆕 每 GPU 分配的 Actor 数量 |
| `param_sync_interval` | `1` | 🆕 多 Learner 参数同步间隔 |

---

## 📊 单机 vs 分布式对比实验

```bash
# 1. 运行单机训练
bash scripts/start.sh  # 选择 2

# 2. 运行分布式训练
bash scripts/start.sh  # 选择 1

# 3. 绘制对比曲线
bash scripts/start.sh  # 选择 4
```

生成 `output/comparison_curves.png`，对比 **训练速度 (SPS)** 和 **平均回报**。

---

## 🌐 Web UI 详细说明

### 开发模式

```bash
bash scripts/start.sh  # 选择 6
```

自动启动：
1. **FastAPI 后端** → http://127.0.0.1:8000
2. **Next.js 开发服务器** → http://localhost:3000
3. API 请求自动代理到 FastAPI

### 生产模式

```bash
cd web && npm install && npm run build && cd ..
bash scripts/start.sh  # 选择 5
# 访问 http://127.0.0.1:8000
```

### 功能列表

| 功能 | 描述 |
|------|------|
| 🎮 训练控制 | 一键启动/停止分布式 & 单机训练 |
| 📡 实时监控 | WebSocket 推送 SPS、回报、损失、Buffer 大小 |
| 📈 在线绘图 | Chart.js 实时训练曲线 |
| ⚖️ 性能对比 | 分布式 vs 单机同屏对比 |
| 🎬 视频生成 | 一键生成智能体表现视频，支持重生成 |

---

## 🦀 Rust 回放缓冲区

<details>
<summary>📖 点击展开详情</summary>

### 特性

- **Structure of Arrays (SoA)** 内存布局 — 连续内存，缓存局部性更优
- **Rayon 并行 GAE** — 多线程并行计算优势函数
- **PyO3 Python 绑定** — 无缝集成到现有 Python 训练代码
- **目标 10-50x** 内存操作速度提升

### 构建

```bash
# 方式一：通过构建脚本
bash scripts/build.sh  # 选择构建 Rust Buffer

# 方式二：手动构建
cd rust_buffer
pip install maturin
maturin develop --release
```

> ⚠️ 需要先安装 Rust：`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

</details>

---

## 🛠️ 技术栈

<table>
<tr>
<th>后端</th>
<th>前端</th>
<th>性能优化</th>
</tr>
<tr>
<td>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Ray](https://img.shields.io/badge/Ray-2.x-028CF0?logo=ray&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3-000000?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHRleHQgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iMTQiIHg9IjIiIHk9IjE4Ij5NPC90ZXh0Pjwvc3ZnPg==)

</td>
<td>

![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?logo=tailwindcss&logoColor=white)
![Zustand](https://img.shields.io/badge/Zustand-5-443E38)
![Chart.js](https://img.shields.io/badge/Chart.js-4-FF6384?logo=chart.js&logoColor=white)

</td>
<td>

![Rust](https://img.shields.io/badge/Rust-1.70+-000000?logo=rust&logoColor=white)
![ndarray](https://img.shields.io/badge/ndarray-0.15-D68A00)
![Rayon](https://img.shields.io/badge/Rayon-1.8-FF6D00)
![PyO3](https://img.shields.io/badge/PyO3-0.20-D64045)

</td>
</tr>
</table>

---

## 🏛️ 架构概览

### 单 GPU 架构（默认）

```
┌─────────────────────────────────────────────────────────┐
│                     Next.js Frontend                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │Distributed│ │  Single  │ │Comparison│ │   Video   │  │
│  │   Tab     │ │   Tab    │ │   Tab    │ │  Section  │  │
│  └─────┬─────┘ └─────┬────┘ └─────┬────┘ └─────┬─────┘  │
│        └──────────────┼───────────┼─────────────┘        │
│                  ┌─────┴─────┐                            │
│                  │  Zustand  │                            │
│                  │   Store   │                            │
│                  └─────┬─────┘                            │
│            ┌───────────┼───────────┐                      │
│       ┌────┴────┐     │     ┌─────┴────┐                │
│       │REST API │     │     │WebSocket │                │
│       └────┬────┘     │     └─────┬────┘                │
└────────────┼──────────┼───────────┼──────────────────────┘
             │          │           │
┌────────────┼──────────┼───────────┼──────────────────────┐
│        ┌───┴──────────┴───────────┴───┐                  │
│        │       FastAPI Backend        │                  │
│        └───────────────┬──────────────┘                  │
│                   ┌────┴────┐                             │
│                   │  Ray    │                             │
│                   │ Cluster │                             │
│        ┌──────────┼────────┼──────────┐                  │
│   ┌────┴────┐ ┌───┴───┐ ┌───┴───┐ ┌──┴──┐              │
│   │ Actor 0 │ │Actor 1│ │Actor N│ │Learner│             │
│   └────┬────┘ └───┬───┘ └───┬───┘ └──┬──┘              │
│        └──────────┼──────────┼────────┘                  │
│              ┌────┴────┐                                  │
│              │  Param  │                                  │
│              │ Server  │                                  │
│              └─────────┘                                  │
│                     Backend (Python)                      │
└──────────────────────────────────────────────────────────┘
```

### 🆕 多 GPU 扩展架构（GPU=4 示例）

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│              /api/scaling/*  /api/cluster/info                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                     Ray Cluster                                │
│                                                                │
│  ┌─ Learner₀ (GPU:0) ←→ [Actor₀..Actor₇]   ← 8 CPU Actors   │
│  ├─ Learner₁ (GPU:1) ←→ [Actor₈..Actor₁₅]                    │
│  ├─ Learner₂ (GPU:2) ←→ [Actor₁₆..Actor₂₃]                   │
│  └─ Learner₃ (GPU:3) ←→ [Actor₂₄..Actor₃₁]                   │
│                                                                │
│  共享: ParameterServer (周期性参数平均)                         │
│  独立: 每个 Learner 有自己的 ReplayBuffer                      │
└────────────────────────────────────────────────────────────────┘
```

每个 Learner 独占 1 块 GPU + 8 个 CPU Actor 采样。通过 ParameterServer 每轮参数平均（Parameter Averaging）保持所有 Learner 的策略一致。

---

## 🖥️ Slurm 集群使用指南

项目提供了完整的 Slurm 集群训练脚本，支持在 GPU 集群上提交分布式/单机训练作业，以及 **🆕 Web UI 交互式训练** 和 **🆕 多 GPU 扩展实验**。

### 第一步：上传项目到集群

```bash
# 在本地机器上，将项目上传到集群
scp -r ./DRL_MuJoCo <用户名>@<集群地址>:~/DRL_MuJoCo
```

### 第二步：SSH 登录集群

```bash
ssh <用户名>@<集群地址>
cd ~/DRL_MuJoCo
```

### 第三步：创建 Conda 环境（仅首次需要）

```bash
# 直接在管理节点上运行（不需要 sbatch）
bash scripts/slurm/setup_env.sh
```

> ⚠️ **运行前先检查集群 CUDA 版本**：
> ```bash
> nvidia-smi  # 查看右上角 CUDA Version
> ```
> 如果是 CUDA 12.1，需先编辑 `scripts/slurm/setup_env.sh` 将 `cu118` 改为 `cu121`。

这一步完成后，`drl_mujoco` Conda 环境和 Next.js 前端就构建好了，**以后不需要再运行**。

### 方式一：Web UI 交互式训练（推荐）

在计算节点上启动 Web UI，通过 SSH 端口转发在本地浏览器中访问：

```bash
# 提交 Web UI 作业（1 GPU 基础模式）
sbatch scripts/slurm/run_webui.sh

# 提交 Web UI 作业（4 GPU 扩展模式）
sbatch --gres=gpu:4 --cpus-per-task=40 --mem=64G scripts/slurm/run_webui.sh

# 查看日志获取连接命令
cat logs/webui_<JOBID>.out
```

日志中会显示 SSH 端口转发命令，在本地终端执行：

```bash
ssh -N -L 8000:<计算节点>:8000 <用户名>@<集群管理节点>
```

然后浏览器打开 **http://localhost:8000** 即可使用 Web UI：
- 🎮 一键启动/停止训练
- 📡 实时监控训练曲线
- 🆕 查看集群 GPU 信息
- 🆕 启动 GPU 扩展实验

### 方式二：命令行提交训练作业

**单独提交一个任务**：

```bash
# 单机模式
sbatch scripts/slurm/run_single.sh

# 分布式模式
sbatch scripts/slurm/run_distributed.sh

# 指定其他环境配置
sbatch scripts/slurm/run_single.sh config/walker2d_single.yaml
sbatch scripts/slurm/run_distributed.sh config/walker2d.yaml
```

**一键批量提交全部 6 个任务**：

```bash
bash scripts/slurm/run_all_envs.sh --dry-run   # 预览
bash scripts/slurm/run_all_envs.sh              # 提交
```

### 🆕 方式三：GPU 扩展性对比实验

支持 4/8/16/32 GPU 的扩展性对比实验：

```bash
# 1. 生成缩放配置（自动缩放 actors、batch_size、buffer）
python scripts/gen_scaling_configs.py --gpu_counts 4 8 16 32

# 2. 预览实验
bash scripts/slurm/run_gpu_experiments.sh --dry-run

# 3. 提交全部实验
bash scripts/slurm/run_gpu_experiments.sh

# 4. 也可以只跑部分实验
bash scripts/slurm/run_gpu_experiments.sh --envs hopper --gpus 4 8

# 5. 分析结果
python scripts/analyze_scaling.py --input_dir output/scaling
```

**GPU 扩展资源缩放对照表**：

| GPU 数量 | 节点数 (每节点 4 GPU) | Actors 总数 | batch_size | CPU 核心 | 内存 |
|:--------:|:--------------------:|:-----------:|:----------:|:--------:|:----:|
| 4 | 1 | 32 | 2048 | 40 | 64G |
| 8 | 2 | 64 | 4096 | 40/节点 | 64G/节点 |
| 16 | 4 | 128 | 8192 | 40/节点 | 64G/节点 |
| 32 | 8 | 256 | 16384 | 40/节点 | 64G/节点 |

> ⚠️ `run_gpu_experiments.sh` 中的 `GPUS_PER_NODE=4` 需根据实际集群修改。

### 监控训练进度

```bash
# 查看所有作业状态
bash scripts/slurm/monitor.sh

# 实时查看某个作业的日志
bash scripts/slurm/monitor.sh <JOB_ID>

# 取消所有作业
bash scripts/slurm/monitor.sh --cancel-all
```

### 训练完成后绘图

```bash
# 绘制全部图表
sbatch scripts/slurm/run_plot.sh

# 或仅绘制特定类型
sbatch scripts/slurm/run_plot.sh --training     # 训练详情图
sbatch scripts/slurm/run_plot.sh --comparison   # 多环境对比图
```

### Slurm 脚本资源分配

| 脚本 | CPUs | 内存 | GPU | 时限 | 说明 |
|:-----|:----:|:----:|:---:|:----:|------|
| `run_single.sh` | 4 | 8G | 1 | 7天 | 单机 1 Actor + 1 Learner |
| `run_distributed.sh` | 12 | 32G | 1 | 7天 | 分布式 8 Actors + Ray 集群 |
| `run_plot.sh` | 2 | 4G | 0 | 1小时 | 纯 CPU 绘图任务 |
| 🆕 `run_webui.sh` | 16 | 32G | 1+ | 7天 | Web UI + 训练（可申请多 GPU） |
| 🆕 `run_scaling.sh` | 16 | 32G | 1+ | 7天 | GPU 扩展实验（支持多节点） |

### 执行流程总结

```
首次使用:
  ① bash scripts/slurm/setup_env.sh            ← 仅需运行一次

Web UI 模式 (交互式):
  ② sbatch scripts/slurm/run_webui.sh          ← 提交 Web UI 作业
  ③ ssh -N -L 8000:<node>:8000 user@gateway    ← 本地端口转发
  ④ 浏览器打开 http://localhost:8000            ← 可视化操作

命令行模式 (后台批量):
  ② sbatch scripts/slurm/run_single.sh         ← 提交单机训练
     或 sbatch scripts/slurm/run_distributed.sh ← 提交分布式训练
     或 bash scripts/slurm/run_all_envs.sh      ← 一键提交全部6个任务

GPU 扩展实验:
  ② python scripts/gen_scaling_configs.py      ← 生成缩放配置
  ③ bash scripts/slurm/run_gpu_experiments.sh  ← 一键提交对比实验

训练中:
  ⑤ bash scripts/slurm/monitor.sh              ← 监控作业

训练后:
  ⑥ sbatch scripts/slurm/run_plot.sh           ← 绘图
  ⑦ python scripts/analyze_scaling.py          ← 🆕 分析 GPU 扩展结果
```

> 日志文件自动写入 `logs/` 目录，格式为 `single_<JOBID>.out`、`dist_<JOBID>.out`、`webui_<JOBID>.out`、`scale_<JOBID>.out`。

### 🆕 Web UI 新增 API

| 端点 | 方法 | 说明 |
|:-----|:----:|------|
| `/api/scaling/configs` | GET | 获取所有 GPU 扩展实验配置 |
| `/api/scaling/metrics?config_name=hopper_gpu4` | GET | 获取扩展实验训练指标 |
| `/api/scaling/start?config_name=hopper_gpu4` | POST | 启动 GPU 扩展实验 |
| `/api/scaling/status` | GET | 获取扩展实验运行状态 |
| `/api/cluster/info` | GET | 获取集群 GPU 信息 + Ray 资源 |
