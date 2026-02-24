from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    env_name: str = "Hopper-v4"
    num_actors: int = 8
    replay_buffer_capacity: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    clip_ratio: float = 0.15
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    rollout_length: int = 2048
    actor_update_interval: int = 1
    learner_updates_per_iter: int = 10
    max_iters: int = 1_000
    log_interval: int = 10
    metrics_path: str = "output/metrics.csv"
    hidden_sizes: tuple[int, int] = (256, 256)
    seed: int = 42
    use_cuda: bool = True
    use_mps: bool = True
    ray_address: str | None = None
