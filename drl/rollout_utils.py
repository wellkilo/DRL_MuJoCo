"""Rollout 阶段使用的纯函数。

该模块不依赖 Ray，便于对 PPO 的关键数学契约做确定性测试。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from drl.running_mean_std import RunningMeanStd

# 使用 Any 避免 Python 3.9 在运行时求值 PEP 604 联合类型；具体字段由函数契约约束。
RMSState = dict[str, Any]


def prepare_action(
    sampled_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回用于 PPO 概率计算的原始动作和实际送入环境的裁剪动作。

    PPO 必须保存采样分布产生的原始动作，否则旧 log-prob 与训练时重新
    计算的 log-prob 不再对应。环境动作可以独立裁剪到合法范围。
    """

    policy_action = np.asarray(sampled_action, dtype=np.float32).copy()
    env_action = np.clip(policy_action, action_low, action_high).astype(np.float32, copy=False)
    return policy_action, env_action


def compute_batch_moments(samples: np.ndarray) -> RMSState:
    """计算一批新观测的 moments，不携带任何历史统计量。"""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("samples must contain at least one batched observation")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples contain NaN or infinite values")
    return {
        "mean": values.mean(axis=0),
        "var": values.var(axis=0),
        "count": float(values.shape[0]),
    }


def merge_rms_states(states: Sequence[dict[str, Any] | None]) -> RMSState | None:
    """合并互不重叠的统计 moments。

    调用方必须传入增量 moments 或彼此独立的完整数据集统计，不能把同一份
    历史数据重复放入多个 state。
    """

    valid_states: list[RMSState] = []
    expected_shape: tuple[int, ...] | None = None
    for state in states:
        if state is None or not {"mean", "var", "count"}.issubset(state):
            continue
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(state["count"])
        if mean.shape != var.shape or count <= 0:
            continue
        if expected_shape is None:
            expected_shape = mean.shape
        if mean.shape != expected_shape:
            raise ValueError("all RMS states must have the same shape")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
            continue
        if np.any(var < 0) or not np.isfinite(count):
            continue
        valid_states.append({"mean": mean, "var": var, "count": count})

    if not valid_states:
        return None

    merged = RunningMeanStd(shape=np.asarray(valid_states[0]["mean"]).shape)
    merged.set_state(valid_states[0])
    for state in valid_states[1:]:
        merged._update_from_moments(
            np.asarray(state["mean"], dtype=np.float64),
            np.asarray(state["var"], dtype=np.float64),
            float(state["count"]),
        )
    return merged.get_state()


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    terminated: Sequence[bool],
    episode_ends: Sequence[bool],
    truncated_bootstrap_values: Sequence[float | None],
    *,
    gamma: float,
    gae_lambda: float,
    last_value: float,
) -> tuple[list[float], list[float]]:
    """计算 GAE，同时正确区分自然终止与 TimeLimit 截断。

    ``terminated`` 控制 value bootstrap；``episode_ends`` 控制 GAE 是否跨
    episode 传播。截断会使用截断前 next observation 的 value bootstrap，
    但不会把下一个 reset episode 的优势传播回来。
    """

    size = len(rewards)
    sequences = (values, terminated, episode_ends, truncated_bootstrap_values)
    if any(len(sequence) != size for sequence in sequences):
        raise ValueError("GAE inputs must have identical lengths")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be within [0, 1]")

    advantages = [0.0] * size
    returns = [0.0] * size
    gae = 0.0

    for index in range(size - 1, -1, -1):
        is_terminated = bool(terminated[index])
        is_episode_end = bool(episode_ends[index])
        if is_terminated and not is_episode_end:
            raise ValueError("terminated transition must also end the episode")

        if is_episode_end:
            if is_terminated:
                next_value = 0.0
            else:
                bootstrap = truncated_bootstrap_values[index]
                if bootstrap is None:
                    raise ValueError("truncated transition requires a bootstrap value")
                next_value = float(bootstrap)
            trace_continuation = 0.0
        else:
            next_value = float(last_value if index == size - 1 else values[index + 1])
            trace_continuation = 1.0

        bootstrap_mask = 0.0 if is_terminated else 1.0
        delta = float(rewards[index]) + gamma * bootstrap_mask * next_value - float(values[index])
        gae = delta + gamma * gae_lambda * trace_continuation * gae
        advantages[index] = gae
        returns[index] = gae + float(values[index])

    return advantages, returns
