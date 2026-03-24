# DRL MuJoCo 分布式训练原型

本项目实现基于 Ray 的分布式 Actor-Learner 架构，面向 MuJoCo 环境进行并行采样与学习。项目包含 TypeScript 前端重写和 Rust 高性能回放缓冲区优化。

## 结果视图体系

### Web UI 总界面
![Web UI 总界面](assets/web_ui.png)
完整的 Web UI 界面，包含训练控制、实时监控、在线绘图和视频演示功能，构成一个完整的训练与结果展示闭环。

### 分布式训练视图
![分布式训练视图](assets/web_ui_distributed.png)
实时监控分布式训练过程的核心指标，包括训练速度（SPS）、平均回报、损失函数和回放池大小等，帮助您了解分布式架构下的训练效率和收敛情况。

### 单机训练视图
![单机训练视图](assets/web_ui_single.png)
与分布式训练视图对应，展示单机模式下的训练指标，为后续的对比分析提供基准数据。通过与分布式视图的横向对比，可以直观评估并行采样带来的性能提升。

### 对比视图
![对比视图](assets/web_ui_comparison.png)
基于分布式和单机训练的历史数据，生成直观的对比曲线，重点展示训练速度（SPS）和平均回报的差异，帮助量化分析分布式训练的优势。

### 视频演示界面
![视频演示界面](assets/web_video_demo.gif)
训练完成后，一键生成分布式和单机训练结果的视频演示，直观展示智能体在 MuJoCo 环境中的表现。两个视频支持独立播放和控制，便于进行策略效果的直观对比。

## 功能概览

- Actor 采样器并行与环境交互
- ReplayBuffer 经验回放
- Learner 进行策略更新
- ParameterServer 分发最新参数
- 支持 CUDA / MPS / CPU 自动选择
- **TypeScript 前端**：React 18 + TypeScript + Zustand + SCSS，类型安全，代码简洁
- **Rust 高性能回放缓冲区**：SoA 内存布局 + 并行 GAE 计算
- **Web UI 可视化界面**：实时监控训练过程，一键启动/停止，在线绘图

## 目录结构

- main.py：训练入口
- drl/：分布式组件与模型实现
- config/：配置文件目录
  - config.yaml：分布式训练配置
  - config_single.yaml：单机训练配置
- output/：输出目录（训练数据、绘图结果）
- scripts/：辅助脚本目录
  - plot_training.py：训练曲线绘制脚本
  - plot_comparison.py：单机与分布式对比曲线绘制脚本
  - build.sh：环境与依赖安装脚本
  - start.sh：训练与绘图启动脚本
- web/：Web UI 目录
  - server.py：FastAPI 后端服务
  - src/：TypeScript 前端源码
    - styles/：SCSS 样式文件
      - _variables.scss：SCSS 变量定义
      - global.scss：全局样式
  - dist/：构建输出目录
- rust_buffer/：Rust 高性能回放缓冲区
  - src/：Rust 源码
    - buffer.rs：回放缓冲区实现（SoA 布局）
    - gae.rs：并行 GAE 计算
    - lib.rs：PyO3 Python 绑定
  - Cargo.toml：Rust 项目配置
  - pyproject.toml：Maturin 配置

## 快速开始

### 1. 环境与依赖安装

```bash
bash scripts/build.sh
```

该脚本会：
- 检查 conda 是否可用
- 创建/激活 `drl-arm` 环境
- 安装所有 Python 依赖
- 可选：构建 TypeScript 前端
- 可选：构建 Rust 回放缓冲区

### 2. 启动训练或绘图

```bash
bash scripts/start.sh
```

该脚本提供交互式菜单，支持：
1. 分布式训练（8 个 Actor）
2. 单机训练（1 个 Actor）
3. 绘制训练曲线
4. 绘制对比曲线
5. 启动 Web UI（FastAPI 后端）
6. 启动 TypeScript Dev Server + Web UI（推荐）

## 配置说明

通过 config/config.yaml 调整超参和运行配置，常用参数如下：

- env_name：MuJoCo 环境
- num_actors：并行采样器数量
- replay_buffer_capacity：回放池容量
- batch_size：训练批次大小
- lr：学习率
- hidden_sizes：策略与价值网络隐藏层大小
- use_cuda / use_mps：设备开关
- metrics_path：指标输出文件路径

## 单机与分布式对比实验

要对比单机训练和分布式训练的效果，可以通过 `bash scripts/start.sh` 依次选择：
1. 运行单机训练
2. 运行分布式训练
3. 绘制对比曲线

生成 `output/comparison_curves.png`，对比：
- 训练速度（SPS）
- 平均回报（收敛速度）

## Web UI 可视化

### 开发模式（推荐）

```bash
bash scripts/start.sh
# 选择选项 6 - Launch TypeScript Dev Server + Web UI
```

这会自动：
1. 启动 FastAPI 后端（http://127.0.0.1:8000）
2. 启动 Vite 开发服务器（http://localhost:5173）
3. Vite 自动代理 /api 和 /ws 请求到 FastAPI 后端

### 生产模式

```bash
cd web
npm run build
bash scripts/start.sh
# 选择选项 5 - Launch Web UI (FastAPI backend)
```

### Web UI 功能

- **开始/停止训练**：一键控制训练进程
- **实时监控**：通过 WebSocket 实时更新训练指标
- **在线绘图**：实时显示训练速度、平均回报、损失、回放池大小曲线
- **性能对比**：分布式 vs 单机训练指标对比
- **一键生成视频**：生成分布式和单机训练结果的视频演示，支持独立播放和控制

## TypeScript 前端特性

- **完整的 TypeScript 类型安全**：所有数据结构都有类型定义
- **React 18 + Zustand**：简洁的状态管理
- **SCSS 样式系统**：变量管理、嵌套选择器、代码复用
- **通用图表组件**：一个组件替代 12 个手动创建的图表，减少代码冗余 70%
- **4 个核心指标展示**：训练速度、平均回报、损失、Buffer 大小
- **WebSocket 实时数据推送**：自动重连机制
- **响应式设计**：适配不同屏幕尺寸

## Rust 回放缓冲区特性

- **Structure of Arrays (SoA) 内存布局**：连续内存，更好的缓存局部性
- **并行 GAE 计算**：使用 Rayon 多线程并行计算优势函数
- **PyO3 Python 绑定**：无缝集成到现有 Python 代码
- **目标性能提升**：10-50x 内存操作速度提升

### 构建 Rust 回放缓冲区

```bash
bash scripts/build.sh
# 选择构建 Rust Buffer
```

或手动构建：

```bash
cd rust_buffer
pip install maturin
maturin develop --release
```

## 技术栈

### 后端
- Python 3.9+
- FastAPI（Web 服务）
- Ray（分布式计算）
- PyTorch（深度学习）
- MuJoCo（物理仿真）

### 前端
- TypeScript
- React 18
- Zustand（状态管理）
- SCSS（样式系统）
- Chart.js（数据可视化）
- Vite（构建工具）

### 性能优化
- Rust + ndarray + Rayon
- PyO3（Python 绑定）
