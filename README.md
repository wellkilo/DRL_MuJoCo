# DRL MuJoCo 分布式训练原型

本项目实现基于 Ray 的分布式 Actor-Learner 架构，面向 MuJoCo 环境进行并行采样与学习。

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
  - templates/：HTML 模板
  - static/：静态资源

## 快速开始

### 1. 环境与依赖安装

```bash
bash scripts/build.sh
```

该脚本会：
- 检查 conda 是否可用
- 创建/激活 `drl-arm` 环境
- 安装所有 Python 依赖

### 2. 启动训练或绘图

```bash
bash scripts/start.sh
```

该脚本提供交互式菜单，支持：
1. 分布式训练（8 个 Actor）
2. 单机训练（1 个 Actor）
3. 绘制训练曲线
4. 绘制对比曲线
5. 启动 Web UI 可视化

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

启动 Web UI 服务器（通过 `bash scripts/start.sh` 选择选项 5），在浏览器中打开：http://127.0.0.1:8000

### Web UI 功能
- **开始/停止训练**：一键控制训练进程
- **实时监控**：通过 WebSocket 实时更新训练指标
- **在线绘图**：实时显示训练速度、平均回报、损失、回放池大小曲线
- **刷新数据**：手动刷新历史训练数据
- **一键生成视频**：生成分布式和单机训练结果的视频演示，支持独立播放和控制
