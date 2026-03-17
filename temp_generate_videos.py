#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from drl.video_generator import generate_video

print("[Video Script] Starting...", flush=True)

# 生成分布式视频
config_path_distributed = str(Path(__file__).parent / "config" / "config.yaml")
output_path_distributed = str(Path(__file__).parent / "output" / "video_distributed.mp4")
print(f"[Video Script] Generating distributed video...", flush=True)
result_dist = generate_video(
    config_path=config_path_distributed,
    output_path=output_path_distributed,
    num_episodes=1,
    max_steps=50,
    fps=30
)
print(f"[Video Script] Distributed result: {result_dist}", flush=True)

# 生成单机视频
config_path_single = str(Path(__file__).parent / "config" / "config_single.yaml")
output_path_single = str(Path(__file__).parent / "output" / "video_single.mp4")
print(f"[Video Script] Generating single video...", flush=True)
result_single = generate_video(
    config_path=config_path_single,
    output_path=output_path_single,
    num_episodes=1,
    max_steps=50,
    fps=30
)
print(f"[Video Script] Single result: {result_single}", flush=True)

print("[Video Script] Done!", flush=True)
