# DRL Ray 分布式组件实现
# 包含分布式训练的核心组件：参数服务器、经验回放池、Actor 和 Learner

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import ray
import torch

from drl.models import ActorCritic


@ray.remote
class ParameterServer:
    """
    参数服务器（Parameter Server）
    功能：
    1. 存储最新的模型参数
    2. 供所有 Actor 和 Learner 异步读写
    3. 实现参数在分布式节点间的同步
    """
    def __init__(self) -> None:
        self._state_dict: dict[str, Any] | None = None

    def get(self) -> dict[str, Any] | None:
        """获取当前存储的模型参数"""
        return self._state_dict

    def set(self, state_dict: dict[str, Any]) -> None:
        """更新存储的模型参数"""
        self._state_dict = state_dict


@ray.remote
class ReplayBuffer:
    """
    经验回放池（Replay Buffer）
    功能：
    1. 存储所有 Actor 采集的经验轨迹
    2. 为 Learner 提供训练数据
    3. 实现经验的重用和解耦
    """
    def __init__(self, capacity: int) -> None:
        # 使用双端队列实现固定容量的经验池
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)

    def add(self, items: list[dict[str, Any]]) -> None:
        """添加一批经验到回放池"""
        self._buffer.extend(items)

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        """
        从回放池随机采样一批经验
        Args:
            batch_size: 采样批次大小
        Returns:
            采样的经验列表
        """
        batch_size = min(batch_size, len(self._buffer))
        if batch_size <= 0:
            return []
        # 随机无放回采样
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def size(self) -> int:
        """返回当前回放池中的经验数量"""
        return len(self._buffer)


def _to_tensor(batch: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """
    将批量的经验数据转换为 PyTorch 张量
    Args:
        batch: 经验数据列表，每条数据包含 obs, act, logp, adv, ret
        device: 目标设备（cuda/mps/cpu）
    Returns:
        字典形式的张量批数据
    """
    obs = torch.as_tensor(np.stack([b["obs"] for b in batch], axis=0), device=device, dtype=torch.float32)
    act = torch.as_tensor(np.stack([b["act"] for b in batch], axis=0), device=device, dtype=torch.float32)
    logp = torch.as_tensor(np.array([b["logp"] for b in batch]), device=device, dtype=torch.float32)
    adv = torch.as_tensor(np.array([b["adv"] for b in batch]), device=device, dtype=torch.float32)
    ret = torch.as_tensor(np.array([b["ret"] for b in batch]), device=device, dtype=torch.float32)
    return {"obs": obs, "act": act, "logp": logp, "adv": adv, "ret": ret}


def _ppo_loss(
    model: ActorCritic,
    batch_t: dict[str, torch.Tensor],
    clip_ratio: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    PPO（Proximal Policy Optimization）损失函数
    包含三部分损失：策略损失（clip）、价值损失、熵损失

    Args:
        model: Actor-Critic 模型
        batch_t: 批量张量数据
        clip_ratio: PPO 裁剪比率，控制策略更新幅度
        vf_coef: 价值损失系数
        ent_coef: 熵系数，鼓励探索

    Returns:
        总损失和指标字典
    """
    obs = batch_t["obs"]
    act = batch_t["act"]
    old_logp = batch_t["logp"]  # 旧策略的对数概率
    adv = batch_t["adv"]          # 优势函数
    ret = batch_t["ret"]          # 回报值

    # 前向传播：获取新策略分布和价值估计
    dist, v = model.get_dist_and_value(obs)
    new_logp = dist.log_prob(act).sum(axis=-1)  # 新策略的对数概率

    # PPO 核心公式：重要性采样比率
    ratio = torch.exp(new_logp - old_logp)

    # 对优势函数进行标准化（减去均值，除以标准差）
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # PPO 裁剪机制：限制策略更新幅度
    clip_adv = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    # 策略损失：取原始比率 * 优势 和 裁剪后 * 优势 的最小值
    policy_loss = -(torch.min(ratio * adv, clip_adv)).mean()

    # 价值损失：MSE 损失
    value_loss = torch.nn.functional.mse_loss(v, ret)

    # 熵损失：最大化熵以保持探索
    entropy_loss = -dist.entropy().sum(axis=-1).mean()

    # 总损失 = 策略损失 + 价值损失系数 * 价值损失 + 熵系数 * 熵损失
    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    # 收集训练指标用于监控
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy": float((-entropy_loss).detach().cpu().item()),
        "ratio": float(ratio.detach().cpu().mean().item()),
    }
    return loss, metrics


@ray.remote
class MuJoCoActor:
    """
    MuJoCo Actor - 环境交互与数据采集器
    功能：
    1. 创建独立的 MuJoCo 环境实例
    2. 与环境交互采集轨迹数据
    3. 计算 GAE（广义优势估计）
    4. 异步并行运行多个实例以提高采样效率
    """
    def __init__(self, env_name: str, actor_id: int, seed: int, hidden_sizes: list[int] | tuple[int, int]) -> None:
        import gymnasium as gym

        self.actor_id = actor_id  # Actor 的唯一标识符
        self.env = gym.make(env_name)  # 创建独立的环境实例

        # 使用种子 + actor_id 确保每个 Actor 有不同的随机种子
        self.rng = np.random.default_rng(seed + actor_id)
        obs, _ = self.env.reset(seed=int(seed + actor_id))

        # 获取环境空间维度
        self.obs_dim = int(np.asarray(obs).shape[0])
        self.action_dim = int(np.asarray(self.env.action_space.shape[0]))

        # 获取动作空间范围（用于裁剪和分布变换）
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self.action_low_t = torch.as_tensor(self.action_low, dtype=torch.float32)
        self.action_high_t = torch.as_tensor(self.action_high, dtype=torch.float32)

        # 创建 Actor-Critic 模型（用于推理）
        self.model = ActorCritic(self.obs_dim, self.action_dim, hidden_sizes=hidden_sizes)
        self.model.eval()  # 设置为评估模式

        self._obs = obs  # 保存当前观测

    def sample(
        self,
        state_dict: dict[str, Any] | None,
        rollout_length: int,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        执行策略采样并计算 GAE
        Args:
            state_dict: 最新的模型参数
            rollout_length: 单次采样轨迹长度
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
        Returns:
            traj: 采样的轨迹数据，包含 obs, act, logp, adv, ret
            stats: 统计信息（回合数、回报等）
        """
        # 更新模型参数（如果是第一次则为 None）
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        # 初始化存储列表
        traj: list[dict[str, Any]] = []  # 轨迹数据
        values: list[float] = []          # 价值函数值
        rewards: list[float] = []         # 奖励
        dones: list[float] = []          # 是否结束

        # 统计信息
        episodes = 0             # 完成的回合数
        episode_return_sum = 0.0  # 累计回报
        episode_len_sum = 0       # 累计长度
        ep_return = 0.0          # 当前回合回报
        ep_len = 0               # 当前回合长度

        # 采样循环：与环境交互收集轨迹
        for _ in range(rollout_length):
            # 将观测转换为张量并增加批次维度
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32).unsqueeze(0)

            # 前向传播：获取策略分布和价值估计（不计算梯度）
            with torch.no_grad():
                dist, value_t = self.model.get_dist_and_value(obs_t)
                # 从策略分布采样动作
                action_t = dist.sample()
                # 计算动作的对数概率
                logp_t = dist.log_prob(action_t).sum(axis=-1)

            # 将动作转换为 numpy 并裁剪到合法范围
            action = action_t.squeeze(0).cpu().numpy()
            action_clipped = np.clip(action, self.action_low, self.action_high)

            # 执行动作并获得环境反馈
            next_obs, reward, terminated, truncated, _ = self.env.step(action_clipped)
            done = bool(terminated or truncated)

            # 保存轨迹数据（存储实际执行的动作）
            traj.append(
                {
                    "obs": np.asarray(self._obs, dtype=np.float32),
                    "act": np.asarray(action_clipped, dtype=np.float32),
                    "logp": float(logp_t.item()),
                }
            )
            values.append(float(value_t.item()))
            rewards.append(float(reward))
            dones.append(float(done))

            # 更新当前回合统计
            ep_return += float(reward)
            ep_len += 1
            self._obs = next_obs

            # 如果回合结束，重置环境
            if done:
                episodes += 1
                episode_return_sum += ep_return
                episode_len_sum += ep_len
                # 使用新的随机种子重置环境
                obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
                self._obs = obs
                ep_return = 0.0
                ep_len = 0

        # 获取最后一个状态的价值估计（用于 GAE 计算）
        with torch.no_grad():
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32).unsqueeze(0)
            _, last_value = self.model.get_dist_and_value(obs_t)
        last_value = float(last_value.item())

        # 如果最后一个状态是终止状态，其价值为 0
        if dones and dones[-1] >= 1.0:
            last_value = 0.0

        # 计算 GAE（广义优势估计）和回报
        advs: list[float] = []  # 优势函数
        rets: list[float] = []  # 回报值
        gae = 0.0  # GAE 累积值

        # 从后向前计算 GAE
        for i in reversed(range(len(traj))):
            # 下一个状态的值
            if dones[i] >= 1.0:
                next_value = 0.0
            else:
                next_value = last_value if i == len(traj) - 1 else values[i + 1]

            # TD 误差
            delta = rewards[i] + gamma * (1.0 - dones[i]) * next_value - values[i]

            # GAE 递归公式
            gae = delta + gamma * gae_lambda * (1.0 - dones[i]) * gae

            # 优势 = GAE，回报 = 优势 + 价值
            adv = gae
            ret = adv + values[i]

            advs.append(adv)
            rets.append(ret)

        # 反转列表使其按时间顺序排列
        advs.reverse()
        rets.reverse()

        # 将优势和回报添加到轨迹数据中
        for idx in range(len(traj)):
            traj[idx]["adv"] = float(advs[idx])
            traj[idx]["ret"] = float(rets[idx])

        # 返回轨迹和统计信息
        stats = {
            "actor_id": self.actor_id,
            "episodes": episodes,
            "episode_return_sum": episode_return_sum,
            "episode_len_sum": episode_len_sum,
        }
        return traj, stats


@ray.remote
class Learner:
    """
    Learner - 学习器
    功能：
    1. 维护 Actor-Critic 模型
    2. 从 ReplayBuffer 采样数据进行训练
    3. 计算 PPO 损失并更新模型参数
    4. 将更新后的参数返回供 ParameterServer 同步
    """
    def __init__(self, obs_dim: int, action_dim: int, replay_buffer: Any, cfg: dict[str, Any]) -> None:
        self.replay_buffer = replay_buffer
        self.cfg = cfg

        # 自动选择计算设备：优先 CUDA，其次 MPS（Apple Silicon），最后 CPU
        self.device = self._select_device()

        # 创建 Actor-Critic 模型
        hidden_sizes = tuple(cfg.get("hidden_sizes", (256, 256)))
        self.model = ActorCritic(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)

        # 创建 Adam 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg["lr"]))
        self.model.train()  # 设置为训练模式

    def _select_device(self) -> torch.device:
        """根据硬件可用性选择最佳计算设备"""
        if torch.cuda.is_available() and self.cfg.get("use_cuda", True):
            return torch.device("cuda")
        if torch.backends.mps.is_available() and self.cfg.get("use_mps", True):
            return torch.device("mps")
        return torch.device("cpu")

    def get_state(self) -> dict[str, Any]:
        """获取当前模型状态字典"""
        return self.model.state_dict()

    def train_step(self, updates: int) -> dict[str, Any]:
        """
        执行多轮训练更新
        Args:
            updates: 训练更新轮数
        Returns:
            包含更新后的模型状态字典和训练指标的字典
        """
        metrics_acc: dict[str, float] = {}  # 累积指标

        # 执行多轮梯度更新
        for _ in range(int(updates)):
            # 从经验回放池采样一个批次
            batch = ray.get(self.replay_buffer.sample.remote(int(self.cfg["batch_size"])))
            if not batch:
                continue

            # 将批量数据转换为张量
            batch_t = _to_tensor(batch, self.device)

            # 计算 PPO 损失和指标
            loss, metrics = _ppo_loss(
                self.model,
                batch_t,
                clip_ratio=float(self.cfg["clip_ratio"]),
                vf_coef=float(self.cfg["vf_coef"]),
                ent_coef=float(self.cfg["ent_coef"]),
            )

            # 反向传播更新参数
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # 累积指标
            for k, v in metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + float(v)

        # 计算平均指标
        if metrics_acc:
            for k in list(metrics_acc.keys()):
                metrics_acc[k] /= float(updates)

        # 返回更新后的模型状态和指标
        return {"state_dict": self.model.state_dict(), "metrics": metrics_acc}
