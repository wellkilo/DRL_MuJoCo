# DRL MuJoCo 分布式训练原型
#
# 模型定义：Actor-Critic 神经网络
# 实现策略网络和价值网络共享底层的架构

from __future__ import annotations

import math

import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform


class ActorCritic(torch.nn.Module):
    """
    Actor-Critic 模型
    架构：
    - 策略网络：观测 -> 隐藏层 -> 动作均值
    - 价值网络：观测 -> 隐藏层 -> 状态价值
    - 使用对数标准差参数表示策略方差
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, int] | list[int] = (256, 256)) -> None:
        """
        Args:
            obs_dim: 观测空间维度
            action_dim: 动作空间维度
            hidden_sizes: 隐藏层大小，必须是两个整数
        """
        super().__init__()
        sizes = tuple(hidden_sizes)
        if len(sizes) != 2:
            raise ValueError("hidden_sizes must contain two integers")

        # 策略网络：输出动作的均值
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, sizes[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.Tanh(),
        )
        self.policy_head = torch.nn.Linear(sizes[1], action_dim)

        # 价值网络：输出状态价值 V(s)
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, sizes[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.Tanh(),
        )
        self.value_head = torch.nn.Linear(sizes[1], 1)

        # 可学习的对数标准差参数（用于构建高斯策略）
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        # 初始化网络权重
        self._init_weights()

    def _init_weights(self) -> None:
        """使用正交初始化方法初始化网络权重"""
        # 策略网络隐藏层：使用 sqrt(2) gain（ReLU/Tanh 的最佳实践）
        for layer in self.policy_net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                torch.nn.init.constant_(layer.bias, 0.0)

        # 价值网络隐藏层：同样使用 sqrt(2) gain
        for layer in self.value_net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                torch.nn.init.constant_(layer.bias, 0.0)

        # 策略网络输出层：使用小 gain (0.01) 避免初始时动作过大
        torch.nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        torch.nn.init.constant_(self.policy_head.bias, 0.0)

        # 价值网络输出层：使用 gain = 1
        torch.nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        torch.nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            obs: 观测张量 [batch_size, obs_dim]
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            value: 状态价值 [batch_size]
        """
        mean = self.policy_head(self.policy_net(obs))
        value = self.value_head(self.value_net(obs)).squeeze(-1)
        return mean, value

    def get_dist_and_value(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        """
        获取策略分布和价值
        Args:
            obs: 观测张量 [batch_size, obs_dim]
        Returns:
            dist: 正态分布（用于采样动作和计算概率）
            value: 状态价值 [batch_size]
        """
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)  # 将对数标准差转换为标准差
        dist = torch.distributions.Normal(mean, std)  # 构建高斯策略分布
        return dist, value

    def get_squashed_dist_and_value(self, obs: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor) -> tuple[TransformedDistribution, torch.Tensor]:
        """
        获取 Tanh-Squash 策略分布和价值
        Args:
            obs: 观测张量 [batch_size, obs_dim]
            action_low: 动作空间下界 [action_dim]
            action_high: 动作空间上界 [action_dim]
        Returns:
            dist: Tanh-Squash 变换分布（用于采样动作和计算概率）
            value: 状态价值 [batch_size]
        """
        mean, value = self.forward(obs)
        log_std = self.log_std.to(mean.device)
        std = torch.exp(log_std)
        base_dist = Normal(mean, std)
        
        # 确保 action_low 和 action_high 在正确的设备上
        action_low = action_low.to(mean.device)
        action_high = action_high.to(mean.device)
        
        # 计算 loc 和 scale
        loc = (action_low + action_high) / 2.0
        scale = (action_high - action_low) / 2.0
        
        transforms = [TanhTransform(), AffineTransform(loc=loc, scale=scale)]
        squashed_dist = TransformedDistribution(base_dist, transforms)
        return squashed_dist, value