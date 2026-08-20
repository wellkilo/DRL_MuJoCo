"""训练编排使用的纯函数与类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TrainingTopology:
    requested_gpus: int
    active_gpus: int
    actors_per_gpu: int

    @property
    def total_actors(self) -> int:
        return self.active_gpus * self.actors_per_gpu


def resolve_training_topology(
    *,
    num_actors: int,
    num_gpus: int,
    actors_per_gpu: int | None,
    available_gpus: int,
) -> TrainingTopology:
    """校验配置并解析当前 Ray 资源下的有效训练拓扑。"""

    if num_actors <= 0:
        raise ValueError("num_actors must be positive")
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if available_gpus < 0:
        raise ValueError("available_gpus cannot be negative")

    if actors_per_gpu is None:
        if num_actors % num_gpus != 0:
            raise ValueError(
                "num_actors must be divisible by num_gpus when actors_per_gpu is omitted"
            )
        resolved_actors_per_gpu = num_actors // num_gpus
    else:
        if actors_per_gpu <= 0:
            raise ValueError("actors_per_gpu must be positive")
        expected_actors = num_gpus * actors_per_gpu
        if num_actors != expected_actors:
            raise ValueError(
                f"num_actors={num_actors} does not match "
                f"num_gpus * actors_per_gpu={expected_actors}"
            )
        resolved_actors_per_gpu = actors_per_gpu

    if available_gpus == 0:
        active_gpus = 1
    else:
        active_gpus = min(num_gpus, available_gpus)

    return TrainingTopology(
        requested_gpus=num_gpus,
        active_gpus=active_gpus,
        actors_per_gpu=resolved_actors_per_gpu,
    )


def average_metrics(metrics_list: list[Mapping[str, float]]) -> dict[str, float]:
    """按实际包含某指标的 Learner 数量计算均值。"""

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: total / counts[key] for key, total in totals.items()}
