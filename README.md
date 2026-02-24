# DRL MuJoCo 分布式训练原型

本项目实现基于 Ray 的分布式 Actor-Learner 架构，面向 MuJoCo 环境进行并行采样与学习。

## 功能概览

- Actor 采样器并行与环境交互
- ReplayBuffer 经验回放
- Learner 进行策略更新
- ParameterServer 分发最新参数
- 支持 CUDA / MPS / CPU 自动选择

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

## 手动使用

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 运行训练

```bash
python main.py
```

训练过程中会输出 `output/metrics.csv`，包含训练速度与回报等指标。

### 绘制曲线

```bash
python scripts/plot_training.py
```

将生成图像文件到 `output/` 目录：

- output/training_curves.png：训练速度、回报、损失、回放池大小曲线

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

要对比单机训练和分布式训练的效果，可以按以下步骤操作：

1. **运行单机训练**
   ```bash
   python main.py config/config_single.yaml
   ```

2. **运行分布式训练**（默认 8 个 Actor）
   ```bash
   python main.py
   ```

3. **绘制训练曲线**
   ```bash
   python scripts/plot_training.py
   ```

4. **绘制对比曲线**
   ```bash
   python scripts/plot_comparison.py
   ```

   生成 `output/comparison_curves.png`，对比：
   - 训练速度（SPS）
   - 平均回报（收敛速度）
