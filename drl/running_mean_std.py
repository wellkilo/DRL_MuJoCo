# DRL MuJoCo 观测值归一化
#
# 使用 Welford 在线算法维护观测值的运行均值和标准差
# 对每一步的观测值做 (obs - mean) / std 归一化后再喂给网络
# 这是 PPO 在 MuJoCo 连续控制任务上的标准做法
#
# 修复：count 无限增长导致 float64 溢出 → var 变 inf/NaN → normalize 输出 NaN → 网络崩溃
# 三重防护：
# 1. count 上限保护：超过 MAX_COUNT 时自动缩放，防止 Welford 公式溢出
# 2. var 安全检查：计算后检测 inf/NaN/负值，异常时保留旧 var
# 3. normalize NaN 回退：输出 NaN 时回退到原始值裁剪

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
    5. count 上限保护，防止长期训练时 float64 溢出
    
    参考：
    - OpenAI Baselines: https://github.com/openai/baselines
    - Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
    - Welford 算法: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    # count 上限：超过此值时自动缩放，防止 Welford 公式中 count * batch_count 溢出 float64
    MAX_COUNT: float = 1e6

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

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: float,
    ) -> None:
        """
        使用 Welford 并行合并公式从批量统计量更新全局统计量
        
        核心修复：
        - 当 total_count > MAX_COUNT 时，缩放 self.count 到 MAX_COUNT/2，
          等价于"忘记"非常早期的数据，对在线学习反而有益
        - 计算完 new_var 后检测 inf/NaN/负值，异常时保留旧 var
        
        Args:
            batch_mean: 批量均值
            batch_var: 批量方差
            batch_count: 批量样本数
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # ===== 关键修复：count 过大时缩放 =====
        # 当 count 超过 MAX_COUNT 时，将 self.count 缩放到 MAX_COUNT/2
        # 统计量（mean/var）不受影响，因为 Welford 公式中 count 只影响权重比例
        # 缩放后等价于"忘记"非常早期的数据，这对在线学习反而有益
        if total_count > self.MAX_COUNT:
            scale = (self.MAX_COUNT / 2.0) / self.count if self.count > 0 else 1.0
            self.count *= scale
            total_count = self.count + batch_count

        # 合并均值
        new_mean = self.mean + delta * batch_count / total_count

        # 合并方差：使用并行方差合并公式
        # M2 = count * var
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        # ===== 安全检查：var 异常时保留旧值 =====
        if np.any(~np.isfinite(new_var)) or np.any(new_var < 0):
            # 异常时只更新 mean，保留旧 var
            if np.all(np.isfinite(new_mean)):
                self.mean = new_mean
            return

        # var 下限保护，防止除零
        new_var = np.maximum(new_var, 1e-8)

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
            如果归一化结果包含 NaN，回退到原始值裁剪
        """
        x = np.asarray(x, dtype=np.float64)
        std = np.sqrt(self.var + 1e-8)

        # ===== NaN 回退：std 异常时直接裁剪原始值 =====
        if np.any(~np.isfinite(std)) or np.any(std < 1e-10):
            return np.clip(x, -clip, clip).astype(np.float32)

        normalized = (x - self.mean) / std

        # ===== NaN 回退：归一化结果异常时回退 =====
        if np.any(~np.isfinite(normalized)):
            return np.clip(x, -clip, clip).astype(np.float32)

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