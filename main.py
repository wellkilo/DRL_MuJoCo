# DRL MuJoCo 分布式训练主程序
# 实现基于 Ray 的 Actor-Learner 架构，支持并行采样和策略更新

from __future__ import annotations

import csv
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch

from drl.config_loader import load_config
from drl.logging_utils import log_event
from drl.ray_components import Learner, MuJoCoActor, ParameterServer, ReplayBuffer

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
            # 获取最终模型状态
            final_state = ray.get(global_state["learner"].get_state.remote())
            print(f"[Main] Got model state, saving...", flush=True)
            
            # 保存模型
            config_name = Path(global_state["config_path"]).stem
            model_path = global_state["OUTPUT_DIR"] / f"model_{config_name}.pt"
            print(f"[Main] Saving to: {model_path}", flush=True)
            torch.save(final_state, model_path)
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
    cfg = load_config(config_path)

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

    env = gym.make(cfg.env_name)
    obs_dim = int(np.asarray(env.observation_space.shape[0]))
    action_dim = int(np.asarray(env.action_space.shape[0]))
    env.close()

    # 初始化 Ray 集群（支持本地或分布式集群）
    ray.init(address=cfg.ray_address) if cfg.ray_address else ray.init()

    # 创建分布式核心组件
    # 1. 经验回放缓冲区 - 存储所有 Actor 采集的经验
    replay_buffer = ReplayBuffer.remote(cfg.replay_buffer_capacity)
    # 2. 参数服务器 - 存储最新模型参数，供所有 Actor 拉取
    param_server = ParameterServer.remote()
    # 3. 学习器 - 负责从 ReplayBuffer 采样并更新模型
    learner = Learner.remote(obs_dim, action_dim, replay_buffer, cfg.__dict__)
    
    # 保存 learner 到全局状态，以便信号处理时可以访问
    global_state["learner"] = learner

    # 初始化：Learner 生成初始模型状态，同步到 ParameterServer
    init_state = ray.get(learner.get_state.remote())
    ray.get(param_server.set.remote(init_state))
    
    # 立即保存初始模型，确保即使训练很快停止也有模型文件
    print(f"[Main] Saving initial model...", flush=True)
    config_name = Path(config_path).stem
    model_path = OUTPUT_DIR / f"model_{config_name}.pt"
    torch.save(init_state, model_path)
    print(f"[Main] Initial model saved to {model_path}", flush=True)

    # 创建多个 Actor 并行采集经验
    actors = [MuJoCoActor.remote(cfg.env_name, i, cfg.seed, cfg.hidden_sizes) for i in range(cfg.num_actors)]

    # 从 ParameterServer 获取初始参数
    init_params = ray.get(param_server.get.remote())

    # 启动所有 Actor 进行第一次采样
    # actor_tasks 字典维护：Ray任务ID -> Actor句柄 的映射
    actor_tasks: dict[ray.ObjectRef, ray.actor.ActorHandle] = {
        actor.sample.remote(init_params, cfg.rollout_length, cfg.gamma, cfg.gae_lambda): actor for actor in actors
    }

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
        ],
    )
    metrics_writer.writeheader()

    # 训练主循环
    start_time = time.time()
    total_steps = 0          # 总采样步数
    total_episodes = 0        # 总回合数
    total_return_sum = 0.0    # 累计回报总和
    best_avg_return = -float('inf')  # 最佳平均回报
    for step in range(cfg.max_iters):
        # 等待至少一个 Actor 完成采样（非阻塞）
        done_ids, _ = ray.wait(list(actor_tasks.keys()), num_returns=len(actor_tasks), timeout=0)

        # 获取当前最新模型参数
        current_params = ray.get(param_server.get.remote())

        # 处理已完成的采样任务
        for done_id in done_ids:
            actor = actor_tasks.pop(done_id)
            traj, stats = ray.get(done_id)  # 获取轨迹和统计信息

            # 将轨迹存入经验回放池
            if traj:
                ray.get(replay_buffer.add.remote(traj))

            # 让该 Actor 使用最新参数继续采样
            actor_tasks[actor.sample.remote(current_params, cfg.rollout_length, cfg.gamma, cfg.gae_lambda)] = actor

            # 累计统计信息
            traj_len = len(traj)
            total_steps += traj_len
            total_episodes += int(stats.get("episodes", 0))
            total_return_sum += float(stats.get("episode_return_sum", 0.0))
            log_event("actor_sample", {"step": step, **stats, "traj_len": traj_len})

        # 当经验池数据量足够时，进行学习更新
        size = ray.get(replay_buffer.size.remote())
        if size >= cfg.batch_size:
            # Learner 进行多轮梯度更新
            train_out = ray.get(learner.train_step.remote(cfg.learner_updates_per_iter))

            # 将更新后的模型参数同步到 ParameterServer
            ray.get(param_server.set.remote(train_out["state_dict"]))

            # 定期记录和保存训练指标
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
                metrics_file.flush()  # 立即写入文件
                
                # 定期保存模型，确保视频生成时能拿到最新的模型
                print(f"[Main] Saving model at step {step}...", flush=True)
                try:
                    current_state = ray.get(learner.get_state.remote())
                    config_name = Path(config_path).stem
                    model_path = OUTPUT_DIR / f"model_{config_name}.pt"
                    torch.save(current_state, model_path)
                    print(f"[Main] Model saved to {model_path}", flush=True)
                    
                    # 检查是否为最佳模型，如果是则保存
                    if not math.isnan(avg_return) and avg_return > best_avg_return:
                        best_avg_return = avg_return
                        global_state["best_avg_return"] = best_avg_return
                        best_model_path = OUTPUT_DIR / f"model_{config_name}_best.pt"
                        torch.save(current_state, best_model_path)
                        print(f"[Main] Best model saved to {best_model_path} with avg_return {best_avg_return:.2f}", flush=True)
                except Exception as e:
                    print(f"[Main] Error saving model: {e}", flush=True)

        # 避免过度占用 CPU
        time.sleep(0.01)

    # 训练结束，保存模型
    final_state = ray.get(learner.get_state.remote())
    config_name = Path(config_path).stem
    model_path = OUTPUT_DIR / f"model_{config_name}.pt"
    torch.save(final_state, model_path)
    print(f"Model saved to {model_path}")
    print(f"Best avg return during training: {best_avg_return:.2f}")
    best_model_path = OUTPUT_DIR / f"model_{config_name}_best.pt"
    if best_model_path.exists():
        print(f"Best model is available at: {best_model_path}")

    # 关闭文件
    metrics_file.close()


if __name__ == "__main__":
    main()
