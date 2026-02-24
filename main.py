from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import ray

from drl.config_loader import load_config
from drl.logging_utils import log_event
from drl.ray_components import Learner, MuJoCoActor, ParameterServer, ReplayBuffer


def main() -> None:
    config_path = "config/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    cfg = load_config(config_path)
    import gymnasium as gym

    env = gym.make(cfg.env_name)
    obs_dim = int(np.asarray(env.observation_space.shape[0]))
    action_dim = int(np.asarray(env.action_space.shape[0]))
    env.close()

    ray.init(address=cfg.ray_address) if cfg.ray_address else ray.init()
    replay_buffer = ReplayBuffer.remote(cfg.replay_buffer_capacity)
    param_server = ParameterServer.remote()
    learner = Learner.remote(obs_dim, action_dim, replay_buffer, cfg.__dict__)
    init_state = ray.get(learner.get_state.remote())
    ray.get(param_server.set.remote(init_state))

    actors = [MuJoCoActor.remote(cfg.env_name, i, cfg.seed, cfg.hidden_sizes) for i in range(cfg.num_actors)]
    init_params = ray.get(param_server.get.remote())
    actor_tasks: dict[ray.ObjectRef, ray.actor.ActorHandle] = {
        actor.sample.remote(init_params, cfg.rollout_length, cfg.gamma, cfg.gae_lambda): actor for actor in actors
    }

    metrics_path = Path(cfg.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("w", newline="")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "step",
            "elapsed_sec",
            "total_steps",
            "sps",
            "episodes",
            "avg_return",
            "buffer_size",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "ratio",
        ],
    )
    metrics_writer.writeheader()

    start_time = time.time()
    total_steps = 0
    total_episodes = 0
    total_return_sum = 0.0
    for step in range(cfg.max_iters):
        done_ids, _ = ray.wait(list(actor_tasks.keys()), num_returns=len(actor_tasks), timeout=0)
        current_params = ray.get(param_server.get.remote())
        for done_id in done_ids:
            actor = actor_tasks.pop(done_id)
            traj, stats = ray.get(done_id)
            if traj:
                ray.get(replay_buffer.add.remote(traj))
            actor_tasks[actor.sample.remote(current_params, cfg.rollout_length, cfg.gamma, cfg.gae_lambda)] = actor
            traj_len = len(traj)
            total_steps += traj_len
            total_episodes += int(stats.get("episodes", 0))
            total_return_sum += float(stats.get("episode_return_sum", 0.0))
            log_event("actor_sample", {"step": step, **stats, "traj_len": traj_len})

        size = ray.get(replay_buffer.size.remote())
        if size >= cfg.batch_size:
            train_out = ray.get(learner.train_step.remote(cfg.learner_updates_per_iter))
            ray.get(param_server.set.remote(train_out["state_dict"]))
            if step % cfg.log_interval == 0:
                elapsed = time.time() - start_time
                sps = total_steps / elapsed if elapsed > 0 else 0.0
                avg_return = total_return_sum / total_episodes if total_episodes > 0 else math.nan
                log_event(
                    "learner_update",
                    {
                        "step": step,
                        "buffer_size": size,
                        "elapsed_sec": elapsed,
                        "total_steps": total_steps,
                        "sps": sps,
                        "episodes": total_episodes,
                        "avg_return": avg_return,
                        **train_out["metrics"],
                    },
                )
                metrics_writer.writerow(
                    {
                        "step": step,
                        "elapsed_sec": elapsed,
                        "total_steps": total_steps,
                        "sps": sps,
                        "episodes": total_episodes,
                        "avg_return": avg_return,
                        "buffer_size": size,
                        "loss": train_out["metrics"].get("loss", math.nan),
                        "policy_loss": train_out["metrics"].get("policy_loss", math.nan),
                        "value_loss": train_out["metrics"].get("value_loss", math.nan),
                        "entropy": train_out["metrics"].get("entropy", math.nan),
                        "ratio": train_out["metrics"].get("ratio", math.nan),
                    }
                )
                metrics_file.flush()
        time.sleep(0.01)

    metrics_file.close()


if __name__ == "__main__":
    main()
