#!/usr/bin/env python3
"""测试脚本：验证 CSV 读取和数据更新逻辑"""

import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent
OUTPUT_DIR = REPO_ROOT / "output"

def load_metrics_file(metrics_path: Path):
    """模拟 server.py 中的 load_metrics_file 函数"""
    if not metrics_path.exists():
        return []
    metrics = []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        print(f"[Test] CSV 表头: {reader.fieldnames}")
        for row in reader:
            metrics.append({k: float(v) if k not in ("step", "episodes") else int(float(v)) for k, v in row.items()})
    return metrics

# 检查是否有 CSV 文件
print("[Test] 检查 output 目录...")
print(f"[Test] metrics.csv 存在: {(OUTPUT_DIR / 'metrics.csv').exists()}")
print(f"[Test] metrics_single.csv 存在: {(OUTPUT_DIR / 'metrics_single.csv').exists()}")

# 测试读取（如果文件存在）
for filename in ["metrics.csv", "metrics_single.csv"]:
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        print(f"\n[Test] 读取 {filename}...")
        data = load_metrics_file(filepath)
        if data:
            print(f"[Test] 读取到 {len(data)} 条数据")
            print(f"[Test] 最新一条数据的字段: {list(data[-1].keys())}")
            print(f"[Test] 最新数据: {data[-1]}")
            # 检查关键字段是否存在
            for key in ["sps", "avg_return", "loss", "buffer_size"]:
                if key in data[-1]:
                    print(f"[Test] ✓ 字段 '{key}' 存在，值: {data[-1][key]}")
                else:
                    print(f"[Test] ✗ 字段 '{key}' 不存在!")
        else:
            print("[Test] 文件存在但没有数据")
