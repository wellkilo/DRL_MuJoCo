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
import numpy as np
import torch
import imageio

from drl.config_loader import load_config
from drl.models import ActorCritic
from drl.running_mean_std import RunningMeanStd


def generate_video(
    config_path: str,
    output_path: str,
    target_duration: float = 15.0,
    fps: int = 30
) -> dict[str, Any]:
    """
    生成训练结果的视频演示 - 固定时长版本
    
    Args:
        config_path: 配置文件路径
        output_path: 输出视频路径
        target_duration: 目标视频时长（秒）
        fps: 视频帧率
    
    Returns:
        包含生成结果信息的字典
    """
    print(f"[Video Generator] Generating video for {config_path}...", flush=True)
    
    # 删除旧的视频文件
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        try:
            output_path_obj.unlink()
            print(f"[Video Generator] Deleted old video: {output_path}", flush=True)
        except Exception as e:
            print(f"[Video Generator] Warning: Could not delete old video: {e}", flush=True)
    
    env = None
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
        # 创建环境 - 设置非常大的 max_episode_steps 防止环境提前终止
        env = gym.make(
            cfg.env_name, 
            render_mode="rgb_array", 
            camera_name="track",
            max_episode_steps=10000  # 足够大的步数限制
        )
        obs_dim = int(np.asarray(env.observation_space.shape[0]))
        action_dim = int(np.asarray(env.action_space.shape[0]))
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
        
        # 先获取一帧以确定渲染分辨率，用于黑帧生成
        frame_shape = None
        try:
            test_frame = env.render()
            if test_frame is not None:
                frame_shape = test_frame.shape
        except Exception:
            pass
        
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
        
        # 观测值归一化统计量
        obs_rms = RunningMeanStd(obs_dim)
        obs_rms_state = None

        if selected_model_path:
            try:
                checkpoint = torch.load(selected_model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "actor" in checkpoint:
                    model.load_state_dict(checkpoint["actor"])
                    # 加载观测归一化统计量
                    if checkpoint.get("obs_rms_state") is not None:
                        obs_rms.set_state(checkpoint["obs_rms_state"])
                        obs_rms_state = checkpoint["obs_rms_state"]
                        print(f"[Video Generator] Obs RMS state loaded from {selected_model_path}", flush=True)
                else:
                    # 兼容旧格式（直接是 state_dict）
                    model.load_state_dict(checkpoint)
                print(f"[Video Generator] Model loaded successfully from {selected_model_path}", flush=True)
            except Exception as e:
                print(f"[Video Generator] Error loading model: {e}", flush=True)
                print(f"[Video Generator] Using random initialization", flush=True)
        
        model.eval()
        
        # 固定帧数录制
        total_frames = int(target_duration * fps)
        print(f"[Video Generator] Recording {total_frames} frames for {target_duration}s video at {fps}fps...", flush=True)
        
        frames = []
        obs, _ = env.reset(seed=cfg.seed)
        
        for frame_idx in range(total_frames):
            if frame_idx % 30 == 0:
                print(f"[Video Generator]   Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)", flush=True)
            
            try:
                # 选择动作（使用归一化后的观测值）
                with torch.no_grad():
                    obs_normalized = obs_rms.normalize(np.asarray(obs, dtype=np.float32))
                    obs_tensor = torch.tensor(obs_normalized, dtype=torch.float32).unsqueeze(0)
                    dist, _ = model.get_dist_and_value(obs_tensor)
                    action = dist.mean.squeeze(0).numpy()
                    # 裁剪动作到合法范围
                    action = np.clip(action, action_low, action_high)
            except Exception as e:
                print(f"[Video Generator]   Frame {frame_idx}: Error selecting action: {e}", flush=True)
                # 使用零动作作为备用
                action = np.zeros_like(action_low)
            
            try:
                # 执行环境步进 - 完全忽略终止标志
                next_obs, reward, _, _, _ = env.step(action)
                # 渲染并保存帧
                frame = env.render()
                frames.append(frame)
                # 更新 observation
                obs = next_obs
            except Exception as e:
                print(f"[Video Generator]   Frame {frame_idx}: Error stepping environment: {e}", flush=True)
                # 出错时重复上一帧
                if frames:
                    frames.append(frames[-1])
                else:
                    # 如果没有帧，创建黑帧 - 使用动态分辨率
                    if frame_shape:
                        frames.append(np.zeros(frame_shape, dtype=np.uint8))
                    else:
                        # 回退到默认分辨率
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                continue
        
        print(f"[Video Generator] Frame collection complete, writing video...", flush=True)
        
        # 使用 imageio 写入视频
        try:
            # 确保帧数正确
            if len(frames) > total_frames:
                frames = frames[:total_frames]
            elif len(frames) < total_frames:
                # 填充最后一帧
                while len(frames) < total_frames:
                    frames.append(frames[-1])
            
            imageio.mimwrite(output_path, frames, fps=fps, quality=8)
            print(f"[Video Generator] Video saved to: {output_path}", flush=True)
            return {
                "success": True,
                "output_path": output_path,
                "duration": target_duration,
                "fps": fps,
                "num_frames": total_frames
            }
        except Exception as write_error:
            print(f"[Video Generator] Error writing video: {write_error}", flush=True)
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to write video: {str(write_error)}"
            }
            
    except Exception as e:
        print(f"[Video Generator] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # 确保环境总是被关闭
        if env is not None:
            try:
                env.close()
                print(f"[Video Generator] Environment closed", flush=True)
            except Exception as close_error:
                print(f"[Video Generator] Warning: Could not close environment: {close_error}", flush=True)


def generate_comparison_videos(
    output_dir: str,
    target_duration: float = 15.0,
    fps: int = 30
) -> dict[str, Any]:
    """
    生成分布式和单机训练的对比视频 - 固定时长版本
    
    Args:
        output_dir: 输出目录
        target_duration: 目标视频时长（秒）
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
        target_duration=target_duration,
        fps=fps
    )
    
    # 生成单机训练的视频
    config_path_single = str(Path(__file__).parent.parent / "config" / "config_single.yaml")
    output_path_single = str(Path(output_dir) / "video_single.mp4")
    print(f"[Video Generator] Generating single video...", flush=True)
    results["single"] = generate_video(
        config_path=config_path_single,
        output_path=output_path_single,
        target_duration=target_duration,
        fps=fps
    )
    
    print("[Video Generator] Comparison video generation complete!", flush=True)
    return results
