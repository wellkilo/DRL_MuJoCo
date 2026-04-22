#!/usr/bin/env python3
"""
gen_scaling_configs.py — 根据 GPU 数量自动生成 YAML 配置文件
为每个 (环境, GPU数量) 组合自动缩放 num_actors, batch_size, buffer 等参数.

Usage:
    python scripts/gen_scaling_configs.py
    python scripts/gen_scaling_configs.py --gpu_counts 4 8 16 32
"""

import argparse
import copy
import os

import yaml

BASE_CONFIGS = {
    "hopper": {
        "env_name": "Hopper-v5",
        "hidden_sizes": [256, 256],
        "batch_size": 512,
        "rollout_length": 2048,
        "learner_updates_per_iter": 10,
        "max_iters": 1000,
        "lr": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.15,
        "vf_coef": 0.5,
        "ent_coef": 0.005,
        "max_grad_norm": 0.5,
        "target_kl": 0.015,
        "lr_schedule": "linear",
        "clip_ratio_value": 0.2,
        "warmup_iters": 50,
        "seed": 42,
        "use_cuda": True,
        "use_mps": False,
        "replay_buffer_capacity": 200000,
    },
    "walker2d": {
        "env_name": "Walker2d-v5",
        "hidden_sizes": [256, 256],
        "batch_size": 512,
        "rollout_length": 2048,
        "learner_updates_per_iter": 10,
        "max_iters": 1000,
        "lr": 0.0001,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.1,
        "vf_coef": 0.5,
        "ent_coef": 0.005,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "lr_schedule": "linear",
        "clip_ratio_value": 0.2,
        "warmup_iters": 50,
        "seed": 42,
        "use_cuda": True,
        "use_mps": False,
        "replay_buffer_capacity": 200000,
    },
    "halfcheetah": {
        "env_name": "HalfCheetah-v5",
        "hidden_sizes": [256, 256],
        "batch_size": 512,
        "rollout_length": 2048,
        "learner_updates_per_iter": 10,
        "max_iters": 1000,
        "lr": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.005,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "lr_schedule": "linear",
        "clip_ratio_value": 0.2,
        "warmup_iters": 50,
        "seed": 42,
        "use_cuda": True,
        "use_mps": False,
        "replay_buffer_capacity": 200000,
    },
}

BASE_NUM_ACTORS = 8
BASE_BATCH_SIZE = 512


def gen_config(env_key: str, num_gpus: int, actors_per_gpu: int = 8) -> dict:
    """根据 GPU 数量生成缩放配置.

    Args:
        env_key: 环境键名 (hopper/walker2d/halfcheetah)
        num_gpus: GPU 数量
        actors_per_gpu: 每 GPU 分配的 Actor 数量

    Returns:
        缩放后的配置字典
    """
    cfg = copy.deepcopy(BASE_CONFIGS[env_key])
    num_actors = num_gpus * actors_per_gpu
    scale = num_actors / BASE_NUM_ACTORS

    cfg["num_gpus"] = num_gpus
    cfg["actors_per_gpu"] = actors_per_gpu
    cfg["num_actors"] = num_actors
    cfg["batch_size"] = int(BASE_BATCH_SIZE * scale)
    cfg["replay_buffer_capacity"] = int(200000 * max(scale, 1))
    cfg["metrics_path"] = f"output/scaling/{env_key}_gpu{num_gpus}/metrics.csv"
    cfg["ray_address"] = None
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 GPU 扩展性实验的 YAML 配置文件"
    )
    parser.add_argument(
        "--gpu_counts", nargs="+", type=int, default=[4, 8, 16, 32],
        help="要测试的 GPU 数量列表"
    )
    parser.add_argument(
        "--actors_per_gpu", type=int, default=8,
        help="每 GPU 分配的 Actor 数量"
    )
    parser.add_argument(
        "--output_dir", default="config/scaling",
        help="配置文件输出目录"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0

    for env_key in BASE_CONFIGS:
        for ng in args.gpu_counts:
            cfg = gen_config(env_key, ng, args.actors_per_gpu)
            path = os.path.join(args.output_dir, f"{env_key}_gpu{ng}.yaml")
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"  [{env_key:12s}] GPU={ng:2d} actors={cfg['num_actors']:3d} "
                  f"batch={cfg['batch_size']:5d} → {path}")
            count += 1

    print(f"\n共生成 {count} 个配置文件 → {args.output_dir}/")


if __name__ == "__main__":
    main()