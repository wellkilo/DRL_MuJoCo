from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import ray
import torch

from drl.models import ActorCritic


@ray.remote
class ParameterServer:
    def __init__(self) -> None:
        self._state_dict: dict[str, Any] | None = None

    def get(self) -> dict[str, Any] | None:
        return self._state_dict

    def set(self, state_dict: dict[str, Any]) -> None:
        self._state_dict = state_dict


@ray.remote
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)

    def add(self, items: list[dict[str, Any]]) -> None:
        self._buffer.extend(items)

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        batch_size = min(batch_size, len(self._buffer))
        if batch_size <= 0:
            return []
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def size(self) -> int:
        return len(self._buffer)


def _to_tensor(batch: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
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
    obs = batch_t["obs"]
    act = batch_t["act"]
    old_logp = batch_t["logp"]
    adv = batch_t["adv"]
    ret = batch_t["ret"]

    dist, v = model.get_dist_and_value(obs)
    new_logp = dist.log_prob(act).sum(axis=-1)
    ratio = torch.exp(new_logp - old_logp)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    clip_adv = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -(torch.min(ratio * adv, clip_adv)).mean()
    value_loss = torch.nn.functional.mse_loss(v, ret)
    entropy_loss = -dist.entropy().sum(axis=-1).mean()
    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

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
    def __init__(self, env_name: str, actor_id: int, seed: int, hidden_sizes: list[int] | tuple[int, int]) -> None:
        import gymnasium as gym

        self.actor_id = actor_id
        self.env = gym.make(env_name)
        self.rng = np.random.default_rng(seed + actor_id)
        obs, _ = self.env.reset(seed=int(seed + actor_id))
        self.obs_dim = int(np.asarray(obs).shape[0])
        self.action_dim = int(np.asarray(self.env.action_space.shape[0]))
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self.model = ActorCritic(self.obs_dim, self.action_dim, hidden_sizes=hidden_sizes)
        self.model.eval()
        self._obs = obs

    def sample(
        self,
        state_dict: dict[str, Any] | None,
        rollout_length: int,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        traj: list[dict[str, Any]] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        episodes = 0
        episode_return_sum = 0.0
        episode_len_sum = 0
        ep_return = 0.0
        ep_len = 0
        for _ in range(rollout_length):
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, value_t = self.model.get_dist_and_value(obs_t)
                action_t = dist.sample()
                logp_t = dist.log_prob(action_t).sum(axis=-1)
            action = action_t.squeeze(0).cpu().numpy()
            clipped = np.clip(action, self.action_low, self.action_high)
            next_obs, reward, terminated, truncated, _ = self.env.step(clipped)
            done = bool(terminated or truncated)
            traj.append(
                {
                    "obs": np.asarray(self._obs, dtype=np.float32),
                    "act": np.asarray(action, dtype=np.float32),
                    "logp": float(logp_t.item()),
                }
            )
            values.append(float(value_t.item()))
            rewards.append(float(reward))
            dones.append(float(done))
            ep_return += float(reward)
            ep_len += 1
            self._obs = next_obs
            if done:
                episodes += 1
                episode_return_sum += ep_return
                episode_len_sum += ep_len
                obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
                self._obs = obs
                ep_return = 0.0
                ep_len = 0
        with torch.no_grad():
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32).unsqueeze(0)
            _, last_value = self.model.get_dist_and_value(obs_t)
        last_value = float(last_value.item())
        if dones and dones[-1] >= 1.0:
            last_value = 0.0
        advs: list[float] = []
        rets: list[float] = []
        gae = 0.0
        for i in reversed(range(len(traj))):
            if dones[i] >= 1.0:
                next_value = 0.0
            else:
                next_value = last_value if i == len(traj) - 1 else values[i + 1]
            delta = rewards[i] + gamma * (1.0 - dones[i]) * next_value - values[i]
            gae = delta + gamma * gae_lambda * (1.0 - dones[i]) * gae
            adv = gae
            ret = adv + values[i]
            advs.append(adv)
            rets.append(ret)
        advs.reverse()
        rets.reverse()
        for idx in range(len(traj)):
            traj[idx]["adv"] = float(advs[idx])
            traj[idx]["ret"] = float(rets[idx])
        stats = {
            "actor_id": self.actor_id,
            "episodes": episodes,
            "episode_return_sum": episode_return_sum,
            "episode_len_sum": episode_len_sum,
        }
        return traj, stats


@ray.remote
class Learner:
    def __init__(self, obs_dim: int, action_dim: int, replay_buffer: Any, cfg: dict[str, Any]) -> None:
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.device = self._select_device()
        hidden_sizes = tuple(cfg.get("hidden_sizes", (256, 256)))
        self.model = ActorCritic(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg["lr"]))
        self.model.train()

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available() and self.cfg.get("use_cuda", True):
            return torch.device("cuda")
        if torch.backends.mps.is_available() and self.cfg.get("use_mps", True):
            return torch.device("mps")
        return torch.device("cpu")

    def get_state(self) -> dict[str, Any]:
        return self.model.state_dict()

    def train_step(self, updates: int) -> dict[str, Any]:
        metrics_acc: dict[str, float] = {}
        for _ in range(int(updates)):
            batch = ray.get(self.replay_buffer.sample.remote(int(self.cfg["batch_size"])))
            if not batch:
                continue
            batch_t = _to_tensor(batch, self.device)
            loss, metrics = _ppo_loss(
                self.model,
                batch_t,
                clip_ratio=float(self.cfg["clip_ratio"]),
                vf_coef=float(self.cfg["vf_coef"]),
                ent_coef=float(self.cfg["ent_coef"]),
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            for k, v in metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + float(v)
        if metrics_acc:
            for k in list(metrics_acc.keys()):
                metrics_acc[k] /= float(updates)
        return {"state_dict": self.model.state_dict(), "metrics": metrics_acc}
