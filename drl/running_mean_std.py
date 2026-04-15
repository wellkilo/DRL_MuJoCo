# DRL MuJoCo 观测值归一化
#
# 使用 Welford 在线算法维护观测值的运行均值和标准差
# 对每一步的观测值做 (obs - mean) / std 归一化后再喂给网络
# 这是 PPO 在 MuJoCo 连续控制任务上的标准做法

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """
    使用 Welford 在线算法计算观测值的运行均值和方差
    
    特点：
    1. 增量更新，不需要存储全部历史数据
    2. 数值稳定，使用 Welford 算法避免精度损失
    3. 支持批量更新（一次传入多个观测值）
    4. 线程安全（每个 Actor 有独立实例）
    
    参考：
    - OpenAI Baselines: https://github.com/openai/baselines
    - Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
    - Welford 算法: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    def __init__(self, shape: tuple[int, ...] | int, epsilon: float = 1e-4) -> None:
        """
        Args:
            shape: 观测值形状（标量用 int，向量用 tuple）
            epsilon: 初始计数，防止除零并提供平滑效果
        """
        if isinstance(shape, int):
            shape = (shape,)
        self.mean: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.var: np.ndarray = np.ones(shape, dtype=np.float64)
        self.count: float = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        使用 Welford 在线算法更新均值和方差
        
        支持两种输入形式：
        - 一维数组 [obs_dim]: 单个观测值
        - 二维数组 [batch, obs_dim]: 批量观测值
        
        Args:
            x: 观测值数组
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            # 单个观测值
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        else:
            # 批量观测值
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = x.shape[0]

        # 合并新旧统计量（Welford 并行合并公式）
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        
        # 合并方差：使用并行方差合并公式
        # M2 = count * var
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """
        对观测值进行归一化
        
        Args:
            x: 原始观测值
            clip: 裁剪范围，防止极端值影响训练稳定性
        
        Returns:
            归一化后的观测值，裁剪到 [-clip, clip]
        """
        x = np.asarray(x, dtype=np.float64)
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def get_state(self) -> dict[str, np.ndarray | float]:
        """获取当前统计状态（用于序列化和同步）"""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }

    def set_state(self, state: dict[str, np.ndarray | float]) -> None:
        """从状态字典恢复统计量（用于同步）"""
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])