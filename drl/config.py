# DRL MuJoCo 配置定义
#
# 使用 dataclass 定义训练超参数和运行配置

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    训练配置类
    包含所有可调超参数和运行配置
    """
    # ==================== 环境配置 ====================
    env_name: str = "Hopper-v4"  # MuJoCo 环境名称

    # ==================== 分布式配置 ====================
    num_actors: int = 8           # 并行 Actor 数量（分布式采样器）
    replay_buffer_capacity: int = 200_000  # 经验回放池容量
    batch_size: int = 256          # 训练批次大小
    rollout_length: int = 2048      # 单次 Actor 采样轨迹长度
    actor_update_interval: int = 1    # Actor 参数更新间隔（步数）
    learner_updates_per_iter: int = 10  # 每次迭代 Learner 更新轮数

    # ==================== PPO 算法超参 ====================
    gamma: float = 0.99           # 折扣因子（GAMMA）
    gae_lambda: float = 0.95       # GAE（广义优势估计）lambda 参数
    clip_ratio: float = 0.15       # PPO 裁剪比率（控制策略更新幅度）
    vf_coef: float = 0.5           # 价值损失系数
    ent_coef: float = 0.005        # 熵系数（鼓励探索）

    # ==================== 训练配置 ====================
    lr: float = 2e-4              # 学习率
    max_iters: int = 1_000         # 最大训练迭代次数
    log_interval: int = 10          # 日志记录间隔（步数）
    metrics_path: str = "output/metrics.csv"  # 训练指标输出路径

    # ==================== 模型配置 ====================
    hidden_sizes: tuple[int, int] = (256, 256)  # 隐藏层大小（策略网络和价值网络）
    seed: int = 42                # 随机种子（用于可复现性）

    # ==================== 设备配置 ====================
    use_cuda: bool = True          # 是否使用 CUDA（GPU）
    use_mps: bool = True          # 是否使用 MPS（Apple Silicon GPU）

    # ==================== Ray 配置 ====================
    ray_address: str | None = None   # Ray 集群地址（None 表示本地模式）

    # ==================== 新增：算法优化配置 ====================
    max_grad_norm: float = 0.5       # 梯度裁剪阈值（防止梯度爆炸）
    target_kl: float = 0.015         # KL 散度早停阈值（防止过度更新）
    lr_schedule: str = "linear"       # 学习率调度：'constant' 或 'linear'
