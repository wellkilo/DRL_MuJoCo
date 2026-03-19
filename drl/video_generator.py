# DRL MuJoCo 视频生成器
#
# 生成训练后的视频演示，支持分布式和单机训练结果的视频对比

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

# 在导入 gymnasium 之前设置环境变量 - 这很重要！
# 在 macOS 上，不设置 MUJOCO_GL，让它自动选择
if "MUJOCO_GL" in os.environ:
    del os.environ["MUJOCO_GL"]
if "PYOPENGL_PLATFORM" in os.environ:
    del os.environ["PYOPENGL_PLATFORM"]

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import imageio

from drl.config_loader import load_config
from drl.models import ActorCritic
from drl.ray_components import Learner


def generate_video(
    config_path: str,
    output_path: str,
    num_episodes: int = 1,
    max_steps: int = 100,
    fps: int = 30
) -> dict[str, Any]:
    """
    生成训练结果的视频演示
    
    Args:
        config_path: 配置文件路径
        output_path: 输出视频路径
        num_episodes: 要录制的回合数
        max_steps: 每回合最大步数
        fps: 视频帧率
    
    Returns:
        包含生成结果信息的字典
    """
    print(f"[Video Generator] Generating video for {config_path}...", flush=True)
    
    # 删除旧的视频文件
    output_path_obj = Path(output_path)
    output_dir_path = output_path_obj.parent
    video_prefix = output_path_obj.stem
    
    # 删除目标视频文件
    if output_path_obj.exists():
        try:
            output_path_obj.unlink()
            print(f"[Video Generator] Deleted old video: {output_path}", flush=True)
        except Exception as e:
            print(f"[Video Generator] Warning: Could not delete old video: {e}", flush=True)
    
    # 删除 RecordVideo 生成的旧视频文件
    for old_video in output_dir_path.glob(f"{video_prefix}-episode-*.mp4"):
        try:
            old_video.unlink()
            print(f"[Video Generator] Deleted old RecordVideo: {old_video}", flush=True)
        except Exception as e:
            print(f"[Video Generator] Warning: Could not delete old RecordVideo: {e}", flush=True)
    try:
        # 先关闭可能存在的 Ray 实例，避免和 MuJoCo 环境冲突
        try:
            import ray
            if ray.is_initialized():
                print(f"[Video Generator] Shutting down existing Ray instance...", flush=True)
                ray.shutdown()
        except Exception:
            pass
        
        cfg = load_config(config_path)
        
        # 初始化环境
        print(f"[Video Generator] Creating environment...", flush=True)
        # 确保输出目录存在
        output_dir_path = Path(output_path).parent
        output_dir_path.mkdir(parents=True, exist_ok=True)
        # 使用 RecordVideo 包装器
        video_prefix = Path(output_path).stem
        env = gym.make(cfg.env_name, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=str(output_dir_path),
            name_prefix=video_prefix,
            episode_trigger=lambda x: True,  # 录制所有回合
            disable_logger=False,
            fps=fps
        )
        obs_dim = int(np.asarray(env.observation_space.shape[0]))
        action_dim = int(np.asarray(env.action_space.shape[0]))
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
        
        # 创建模型
        print(f"[Video Generator] Creating model...", flush=True)
        model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=cfg.hidden_sizes
        )
        
        # 尝试加载模型 - 优先加载最佳模型
        config_name = Path(config_path).stem
        best_model_path = Path(output_path).parent / f"model_{config_name}_best.pt"
        model_path = Path(output_path).parent / f"model_{config_name}.pt"
        
        # 优先尝试加载最佳模型
        selected_model_path = None
        if best_model_path.exists():
            selected_model_path = best_model_path
            print(f"[Video Generator] Found best model at {best_model_path}, using it", flush=True)
        elif model_path.exists():
            selected_model_path = model_path
            print(f"[Video Generator] Best model not found, using regular model at {model_path}", flush=True)
        else:
            print(f"[Video Generator] No model files found, using random initialization", flush=True)
        
        if selected_model_path:
            try:
                state_dict = torch.load(selected_model_path, map_location="cpu")
                if "actor" in state_dict:
                    model.load_state_dict(state_dict["actor"])
                else:
                    model.load_state_dict(state_dict)
                print(f"[Video Generator] Model loaded successfully from {selected_model_path}", flush=True)
            except Exception as e:
                print(f"[Video Generator] Error loading model: {e}", flush=True)
                print(f"[Video Generator] Using random initialization", flush=True)
        
        model.eval()
        
        # 录制视频 - RecordVideo 会自动处理
        for episode in range(num_episodes):
            print(f"[Video Generator] Recording episode {episode + 1}/{num_episodes}", flush=True)
            try:
                obs, _ = env.reset(seed=cfg.seed + episode)
                print(f"[Video Generator]   Environment reset done", flush=True)
            except Exception as e:
                print(f"[Video Generator]   Error resetting environment: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
                
            for step in range(max_steps):
                if step % 10 == 0:
                    print(f"[Video Generator]   Step {step}/{max_steps}", flush=True)
                
                try:
                    # 选择动作
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        dist, _ = model.get_dist_and_value(obs_tensor)
                        action = dist.mean.squeeze(0).numpy()
                        # 裁剪动作到合法范围
                        action = np.clip(action, action_low, action_high)
                except Exception as e:
                    print(f"[Video Generator]   Step {step}: Error selecting action: {e}", flush=True)
                    # 使用零动作作为备用
                    action = np.zeros_like(action_low)
                
                try:
                    # 执行环境步进 - RecordVideo 会自动渲染和录制
                    obs, reward, terminated, truncated, _ = env.step(action)
                except Exception as e:
                    print(f"[Video Generator]   Step {step}: Error stepping environment: {e}", flush=True)
                    break
                
                # 即使环境终止，我们也继续运行，直到达到 max_steps
                if terminated or truncated:
                    print(f"[Video Generator]   Episode terminated at step {step}, continuing to run to max_steps...", flush=True)
                    # 继续运行但不重置环境，确保达到完整的步数
                    # 保持 terminated 状态但继续执行
            
            print(f"[Video Generator] Episode {episode + 1} completed", flush=True)
        
        env.close()
        
        # 查找 RecordVideo 生成的视频文件
        print(f"[Video Generator] Looking for generated video...", flush=True)
        generated_video_path = None
        output_dir_path = Path(output_path).parent
        video_prefix = Path(output_path).stem
        # 查找最新生成的视频文件
        video_files = sorted(
            output_dir_path.glob(f"{video_prefix}-episode-*.mp4"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if video_files:
            generated_video_path = video_files[0]
            print(f"[Video Generator] Found generated video: {generated_video_path}", flush=True)
            # 重命名文件到目标路径
            try:
                generated_video_path.rename(output_path)
                print(f"[Video Generator] Video renamed to: {output_path}", flush=True)
                return {
                    "success": True,
                    "output_path": output_path,
                    "num_episodes": num_episodes
                }
            except Exception as rename_error:
                print(f"[Video Generator] Error renaming video: {rename_error}", flush=True)
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": f"Failed to rename video: {str(rename_error)}"
                }
        else:
            print(f"[Video Generator] No video generated by RecordVideo", flush=True)
            return {
                "success": False,
                "error": "No video generated by RecordVideo"
            }
            
    except Exception as e:
        print(f"[Video Generator] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def generate_comparison_videos(
    output_dir: str,
    num_episodes: int = 1,
    max_steps: int = 100,
    fps: int = 30
) -> dict[str, Any]:
    """
    生成分布式和单机训练的对比视频
    
    Args:
        output_dir: 输出目录
        num_episodes: 要录制的回合数
        max_steps: 每回合最大步数
        fps: 视频帧率
    
    Returns:
        包含生成结果信息的字典
    """
    print("[Video Generator] Starting comparison video generation...", flush=True)
    
    results = {}
    
    # 生成分布式训练的视频
    config_path_distributed = str(Path(__file__).parent.parent / "config" / "config.yaml")
    output_path_distributed = str(Path(output_dir) / "video_distributed.mp4")
    print(f"[Video Generator] Generating distributed video...", flush=True)
    results["distributed"] = generate_video(
        config_path=config_path_distributed,
        output_path=output_path_distributed,
        num_episodes=num_episodes,
        max_steps=max_steps,
        fps=fps
    )
    
    # 生成单机训练的视频
    config_path_single = str(Path(__file__).parent.parent / "config" / "config_single.yaml")
    output_path_single = str(Path(output_dir) / "video_single.mp4")
    print(f"[Video Generator] Generating single video...", flush=True)
    results["single"] = generate_video(
        config_path=config_path_single,
        output_path=output_path_single,
        num_episodes=num_episodes,
        max_steps=max_steps,
        fps=fps
    )
    
    print("[Video Generator] Comparison video generation complete!", flush=True)
    return results
