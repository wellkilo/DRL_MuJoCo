# DRL MuJoCo 分布式训练主程序
# 实现基于 Ray 的 Actor-Learner 架构，支持并行采样和策略更新
# 支持多 GPU 扩展：每个 GPU 运行一个 Learner，通过 ParameterServer 参数平均保持策略一致

from __future__ import annotations

import csv
import math
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

# 设置Ray环境变量以消除FutureWarning和metrics警告
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_metrics_export_port"] = "0"

import numpy as np
import ray
import torch

from drl.config_loader import load_config
from drl.logging_utils import log_event
from drl.ray_components import Learner, MuJoCoActor, ParameterServer, ReplayBuffer, _merge_obs_rms_states

# 全局变量用于信号处理
global_state: dict[str, Any] = {}


def save_model_and_exit(signum: int, frame: Any) -> None:
    """
    信号处理函数：在收到终止信号时保存模型并退出
    """
    print("\n[Main] Received termination signal, saving model...", flush=True)
    print(f"[Main] Global state keys: {list(global_state.keys())}", flush=True)
    
    try:
        if "learner" in global_state and "config_path" in global_state and "OUTPUT_DIR" in global_state:
            print(f"[Main] Getting model state from learner...", flush=True)
            # 获取最终模型状态（使用第一个 Learner 作为代表）
            final_state = ray.get(global_state["learner"].get_state.remote())
            print(f"[Main] Got model state, saving...", flush=True)
            
            # 获取 obs_rms 统计量
            obs_rms_state = None
            if "param_server" in global_state:
                try:
                    obs_rms_state = ray.get(global_state["param_server"].get_obs_rms.remote())
                except Exception:
                    pass
            
            # 保存模型（包含 obs_rms 统计量）
            config_name = Path(global_state["config_path"]).stem
            model_path = global_state["OUTPUT_DIR"] / f"model_{config_name}.pt"
            print(f"[Main] Saving to: {model_path}", flush=True)
            torch.save({"actor": final_state, "obs_rms_state": obs_rms_state}, model_path)
            print(f"[Main] Model saved to {model_path}", flush=True)
            
            # 打印最佳模型信息
            if "best_avg_return" in global_state:
                print(f"[Main] Best avg return during training: {global_state['best_avg_return']:.2f}", flush=True)
                best_model_path = global_state["OUTPUT_DIR"] / f"model_{config_name}_best.pt"
                if best_model_path.exists():
                    print(f"[Main] Best model is available at: {best_model_path}", flush=True)
        else:
            print(f"[Main] Missing required keys in global state", flush=True)
    except Exception as e:
        print(f"[Main] Error saving model: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    # 关闭 metrics 文件
    if "metrics_file" in global_state:
        try:
            global_state["metrics_file"].close()
            print("[Main] Metrics file closed", flush=True)
        except Exception:
            pass
    
    # 清理 Ray
    try:
        ray.shutdown()
        print("[Main] Ray shutdown", flush=True)
    except Exception:
        pass
    
    print("[Main] Exiting...", flush=True)
    sys.exit(0)


def main() -> None:
    # 加载配置文件（默认路径或命令行指定）
    config_path = "config/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        cfg = load_config(config_path)
    except Exception as e:
        print(f"[Main] FATAL: Failed to load config from '{config_path}': {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 定义输出目录
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 清理旧的 metrics 文件，避免字段不匹配的问题
    metrics_path = Path(cfg.metrics_path)
    if metrics_path.exists():
        print(f"[Main] Removing old metrics file: {metrics_path}", flush=True)
        metrics_path.unlink()
    
    # 保存全局状态
    global_state["config_path"] = config_path
    global_state["OUTPUT_DIR"] = OUTPUT_DIR
    global_state["best_avg_return"] = -float('inf')
    
    # 注册信号处理函数
    signal.signal(signal.SIGINT, save_model_and_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, save_model_and_exit)  # Terminate signal

    # 导入 Gymnasium 并初始化环境以获取观测和动作空间维度
    import gymnasium as gym

    try:
        env = gym.make(cfg.env_name)
        obs_dim = int(np.asarray(env.observation_space.shape[0]))
        action_dim = int(np.asarray(env.action_space.shape[0]))
        env.close()
    except Exception as e:
        print(f"[Main] FATAL: Failed to create environment '{cfg.env_name}': {e}", flush=True)
        print(f"[Main] Make sure MuJoCo is properly installed. Try: pip install mujoco", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---- Ray 初始化 (支持集群模式) ----
    try:
        if cfg.ray_address:
            ray.init(address=cfg.ray_address)
        elif os.environ.get("RAY_ADDRESS"):
            ray.init(address=os.environ["RAY_ADDRESS"])
        else:
            ray.init()
    except Exception as e:
        print(f"[Main] FATAL: Failed to initialize Ray: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"[Main] Ray cluster resources: {ray.cluster_resources()}", flush=True)

    # ---- 根据 num_gpus 创建多个 Learner ----
    num_gpus = getattr(cfg, 'num_gpus', 1)
    actors_per_gpu = getattr(cfg, 'actors_per_gpu', 8)
    param_sync_interval = getattr(cfg, 'param_sync_interval', 1)

    # 检测 Ray 集群中可用的 GPU 数量, 自动调整 num_gpus
    ray_gpu_count = int(ray.cluster_resources().get("GPU", 0))
    if ray_gpu_count > 0 and num_gpus > ray_gpu_count:
        print(f"[Main] WARNING: Requested {num_gpus} GPUs but only {ray_gpu_count} available. "
              f"Adjusting num_gpus={ray_gpu_count}", flush=True)
        num_gpus = ray_gpu_count
    elif ray_gpu_count == 0 and num_gpus > 1:
        print(f"[Main] WARNING: No GPUs in Ray cluster, falling back to single Learner (CPU/MPS)", flush=True)
        num_gpus = 1

    num_actors_total = num_gpus * actors_per_gpu

    print(f"[Main] Configuration: {num_gpus} GPUs × {actors_per_gpu} Actors/GPU = "
          f"{num_actors_total} total Actors", flush=True)

    # 共享 ParameterServer
    param_server = ParameterServer.remote()
    global_state["param_server"] = param_server
    
    # 创建 K 个 Learner, 每个占 1 GPU + 自己的 ReplayBuffer
    # 当 Ray 集群有 GPU 时, 使用 Learner.options(num_gpus=1).remote() 请求 GPU 资源
    # 当没有 GPU 时, 使用 Learner.remote() 在 CPU/MPS 上运行
    learners = []
    buffers = []
    for g in range(num_gpus):
        buf = ReplayBuffer.remote(cfg.replay_buffer_capacity)
        if ray_gpu_count > 0:
            learner = Learner.options(num_gpus=1).remote(obs_dim, action_dim, buf, cfg.__dict__)
        else:
            learner = Learner.remote(obs_dim, action_dim, buf, cfg.__dict__)
        learners.append(learner)
        buffers.append(buf)
    
    global_state["learner"] = learners[0]  # 信号处理用第一个 Learner
    
    # 初始化: 用第一个 Learner 的参数同步到 PS
    init_state = ray.get(learners[0].get_state.remote())
    ray.get(param_server.set.remote(init_state))
    
    # 将初始参数同步到所有 Learner
    for learner in learners[1:]:
        ray.get(learner.set_state.remote(init_state))
    
    # 立即保存初始模型，确保即使训练很快停止也有模型文件
    print(f"[Main] Saving initial model...", flush=True)
    config_name = Path(config_path).stem
    model_path = OUTPUT_DIR / f"model_{config_name}.pt"
    torch.save({"actor": init_state, "obs_rms_state": None}, model_path)
    print(f"[Main] Initial model saved to {model_path}", flush=True)

    # 创建 Actors: 每个 Learner 分配 actors_per_gpu 个
    all_actors = []
    actor_to_learner = {}   # actor handle id → learner_index 映射
    actor_to_buffer = {}    # actor handle id → buffer 映射
    for g in range(num_gpus):
        for a in range(actors_per_gpu):
            actor_id = g * actors_per_gpu + a
            actor = MuJoCoActor.remote(cfg.env_name, actor_id, cfg.seed, cfg.hidden_sizes)
            all_actors.append(actor)
            actor_to_learner[id(actor)] = g
            actor_to_buffer[id(actor)] = buffers[g]

    print(f"[Main] Created {num_gpus} Learners × {actors_per_gpu} Actors = "
          f"{num_actors_total} total Actors", flush=True)

    # 从 ParameterServer 获取初始参数
    init_params = ray.get(param_server.get.remote())

    # 获取初始观测归一化统计量（首次为 None）
    init_obs_rms = ray.get(param_server.get_obs_rms.remote())

    # 启动所有 Actor 进行第一次采样
    actor_tasks: dict[ray.ObjectRef, ray.actor.ActorHandle] = {}
    for actor in all_actors:
        task = actor.sample.remote(init_params, cfg.rollout_length, cfg.gamma, cfg.gae_lambda, init_obs_rms)
        actor_tasks[task] = actor

    # 准备训练指标输出文件
    metrics_path = Path(cfg.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("w", newline="")
    
    # 保存 metrics_file 到全局状态
    global_state["metrics_file"] = metrics_file
    
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "step",        # 训练步数
            "elapsed_sec", # 已用时间（秒）
            "total_steps", # 总采样步数
            "sps",        # 每秒采样步数（Samples Per Second）
            "episodes",   # 完成回合数
            "avg_return", # 平均回合回报
            "buffer_size", # 经验回放池大小
            "loss",       # 总损失
            "policy_loss", # 策略损失
            "value_loss",  # 价值损失
            "entropy",     # 策略熵（探索程度）
            "ratio",       # PPO 重要性采样比率
            "approx_kl",   # 近似 KL 散度
            "clip_fraction", # 被裁剪的比例
            "explained_var", # 价值函数的解释方差
            "grad_norm",   # 梯度范数
            "lr",          # 当前学习率
            "num_gpus",    # GPU 数量
        ],
    )
    metrics_writer.writeheader()

    # 训练主循环 - 多 Learner 并行训练
    start_time = time.time()
    total_steps = 0          # 总采样步数
    total_episodes = 0        # 总回合数
    total_return_sum = 0.0    # 累计回报总和
    recent_returns = deque(maxlen=100)  # 滑动窗口追踪最近 100 个 episode 的回报
    best_avg_return = -float('inf')  # 最佳平均回报
    
    for train_step in range(cfg.max_iters):
        # 阶段1：等待所有 Actor 完成采样（同步收集）
        print(f"[Main] Train step {train_step}: Collecting samples from "
              f"{num_actors_total} Actors...", flush=True)
        done_ids, _ = ray.wait(list(actor_tasks.keys()), num_returns=len(actor_tasks))

        # 批量获取所有结果
        results = ray.get(done_ids)
        
        # 收集所有 Actor 的 obs_rms 统计量
        all_obs_rms_states: list[dict] = []

        # 处理所有已完成的采样任务
        for done_id, (traj, stats) in zip(done_ids, results):
            actor = actor_tasks.pop(done_id)
            buf = actor_to_buffer[id(actor)]

            # 将轨迹存入对应的 ReplayBuffer
            if traj:
                ray.get(buf.add.remote(traj))

            # 收集 Actor 的观测归一化统计量
            if "obs_rms_state" in stats:
                all_obs_rms_states.append(stats["obs_rms_state"])

            # 累计统计信息
            traj_len = len(traj)
            total_steps += traj_len
            ep_count = int(stats.get("episodes", 0))
            ep_return_sum = float(stats.get("episode_return_sum", 0.0))
            total_episodes += ep_count
            total_return_sum += ep_return_sum
            
            # 将本轮 episode 的平均回报加入滑动窗口
            if ep_count > 0:
                recent_returns.append(ep_return_sum / ep_count)
                
            # 排除 obs_rms_state（包含 numpy 数组，不可 JSON 序列化）
            log_stats = {k: v for k, v in stats.items() if k != "obs_rms_state"}
            log_event("actor_sample", {"step": train_step, **log_stats, "traj_len": traj_len})

        # 合并所有 Actor 的 obs_rms 统计量并同步到 ParameterServer
        if all_obs_rms_states:
            merged_obs_rms = _merge_obs_rms_states(all_obs_rms_states)
            if merged_obs_rms is not None:
                ray.get(param_server.update_obs_rms.remote(merged_obs_rms))

        # 阶段2：所有 Learner 并行训练
        train_futures = [
            learner.train_step.remote(cfg.learner_updates_per_iter)
            for learner in learners
        ]
        train_results = ray.get(train_futures)  # 所有 Learner 并行训练完成

        # 阶段3：参数平均 → 同步到 PS → 同步到所有 Learner
        all_state_dicts = [tr["state_dict"] for tr in train_results]
        
        if num_gpus > 1 and (train_step % param_sync_interval == 0):
            # 多 Learner: 参数平均
            avg_params = ray.get(
                param_server.average_and_set.remote(all_state_dicts)
            )
            # 将平均后的参数同步回每个 Learner
            sync_futures = [
                learner.set_state.remote(avg_params) for learner in learners
            ]
            ray.get(sync_futures)
        else:
            # 单 Learner 或非同步轮: 直接设置
            avg_params = all_state_dicts[0]
            ray.get(param_server.set.remote(avg_params))

        # ===== 关键优化：先发起 Actor 采样，再做保存等操作 =====
        current_obs_rms = ray.get(param_server.get_obs_rms.remote())
        actor_tasks = {}
        for actor in all_actors:
            task = actor.sample.remote(avg_params, cfg.rollout_length, cfg.gamma, cfg.gae_lambda, current_obs_rms)
            actor_tasks[task] = actor

        # ===== 阶段4：日志和模型保存（此时 Actor 已经在并行采样） =====
        # 使用第一个 Learner 的 metrics 作为代表
        metrics_0 = train_results[0]["metrics"]
        elapsed = time.time() - start_time
        sps = total_steps / elapsed if elapsed > 0 else 0.0
        avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else math.nan
        
        # 获取总 buffer 大小
        total_buffer_size = 0
        for buf in buffers:
            total_buffer_size += ray.get(buf.size.remote())

        log_event(
            "learner_update",
            {
                "step": train_step,
                "buffer_size": total_buffer_size,
                "elapsed_sec": elapsed,
                "total_steps": total_steps,
                "sps": sps,
                "episodes": total_episodes,
                "avg_return": avg_return,
                **metrics_0,
            },
        )
        metrics_writer.writerow(
            {
                "step": train_step,
                "elapsed_sec": elapsed,
                "total_steps": total_steps,
                "sps": sps,
                "episodes": total_episodes,
                "avg_return": avg_return,
                "buffer_size": total_buffer_size,
                "loss": metrics_0.get("loss", math.nan),
                "policy_loss": metrics_0.get("policy_loss", math.nan),
                "value_loss": metrics_0.get("value_loss", math.nan),
                "entropy": metrics_0.get("entropy", math.nan),
                "ratio": metrics_0.get("ratio", math.nan),
                "approx_kl": metrics_0.get("approx_kl", math.nan),
                "clip_fraction": metrics_0.get("clip_fraction", math.nan),
                "explained_var": metrics_0.get("explained_var", math.nan),
                "grad_norm": metrics_0.get("grad_norm", math.nan),
                "lr": metrics_0.get("lr", math.nan),
                "num_gpus": num_gpus,
            }
        )
        metrics_file.flush()  # 立即写入文件
        
        # 保存模型（直接用 avg_params，无需再 ray.get(learner.get_state)）
        try:
            model_path = OUTPUT_DIR / f"model_{config_name}.pt"
            torch.save({"actor": avg_params, "obs_rms_state": current_obs_rms}, model_path)
            
            # 检查是否为最佳模型，如果是则保存
            if not math.isnan(avg_return) and avg_return > best_avg_return:
                best_avg_return = avg_return
                global_state["best_avg_return"] = best_avg_return
                best_model_path = OUTPUT_DIR / f"model_{config_name}_best.pt"
                torch.save({"actor": avg_params, "obs_rms_state": current_obs_rms}, best_model_path)
                print(f"[Main] Best model: avg_return={best_avg_return:.2f}", flush=True)
        except Exception as e:
            print(f"[Main] Error saving model: {e}", flush=True)

        # 打印日志（减少频率）
        if train_step % 10 == 0:
            print(
                f"[Main] Step {train_step}/{cfg.max_iters} | "
                f"avg_return={avg_return:.1f} | steps={total_steps:,} | SPS={sps:.0f} | "
                f"GPUs={num_gpus}",
                flush=True
            )

    # 训练结束，保存模型（包含 obs_rms 统计量）
    final_state = ray.get(learners[0].get_state.remote())
    final_obs_rms = ray.get(param_server.get_obs_rms.remote())
    config_name = Path(config_path).stem
    model_path = OUTPUT_DIR / f"model_{config_name}.pt"
    torch.save({"actor": final_state, "obs_rms_state": final_obs_rms}, model_path)
    print(f"Model saved to {model_path}")
    print(f"Best avg return during training: {best_avg_return:.2f}")
    best_model_path = OUTPUT_DIR / f"model_{config_name}_best.pt"
    if best_model_path.exists():
        print(f"Best model is available at: {best_model_path}")

    # 关闭文件
    metrics_file.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Main] FATAL: Unhandled exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
