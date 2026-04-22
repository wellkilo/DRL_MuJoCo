#!/usr/bin/env python3
"""
analyze_scaling.py — GPU 扩展性实验结果分析
读取 output/scaling/ 下各实验的 metrics.csv, 生成汇总对比表.

Usage:
    python scripts/analyze_scaling.py --input_dir output/scaling
    python scripts/analyze_scaling.py --input_dir output/scaling --gpus 4 8 16 32
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Optional


def load_csv(path: str) -> list[dict[str, str]]:
    """加载 CSV 文件并返回行列表."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def analyze(path: str) -> Optional[dict]:
    """分析单个实验的 metrics.csv, 返回汇总统计.

    Args:
        path: metrics.csv 文件路径

    Returns:
        包含 peak_return, final_return, mean_sps, time_min, total_steps 的字典,
        或 None (如果文件不存在或为空)
    """
    if not os.path.exists(path):
        return None
    rows = load_csv(path)
    if not rows:
        return None

    returns = [float(r["avg_return"]) for r in rows
               if r.get("avg_return") and not math.isnan(float(r["avg_return"]))]
    sps_vals = [float(r["sps"]) for r in rows if r.get("sps")]
    final_50 = rows[-50:]
    final_ret = [float(r["avg_return"]) for r in final_50
                 if r.get("avg_return") and not math.isnan(float(r["avg_return"]))]
    elapsed = float(rows[-1].get("elapsed_sec", 0))

    return {
        "peak_return": max(returns) if returns else float("nan"),
        "final_return": sum(final_ret) / len(final_ret) if final_ret else float("nan"),
        "mean_sps": sum(sps_vals) / len(sps_vals) if sps_vals else 0,
        "time_min": elapsed / 60.0,
        "total_steps": int(float(rows[-1].get("total_steps", 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU 扩展性实验结果分析"
    )
    parser.add_argument(
        "--input_dir", default="output/scaling",
        help="实验结果目录"
    )
    parser.add_argument(
        "--envs", nargs="+", default=["hopper", "walker2d", "halfcheetah"],
        help="要分析的环境列表"
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[4, 8, 16, 32],
        help="要分析的 GPU 数量列表"
    )
    args = parser.parse_args()

    results: dict[str, dict[int, dict]] = defaultdict(dict)

    for env in args.envs:
        for g in args.gpus:
            path = os.path.join(args.input_dir, f"{env}_gpu{g}", "metrics.csv")
            res = analyze(path)
            if res:
                results[env][g] = res

    if not results:
        print(f"未找到实验结果，请检查目录: {args.input_dir}")
        return

    print(f"\n{'='*85}")
    print(f"  GPU Scaling Experiment Results")
    print(f"{'='*85}")

    for env in args.envs:
        if env not in results:
            continue
        print(f"\n{'─'*85}\n  {env.upper()}\n{'─'*85}")
        print(f"{'GPU':>5} | {'Actors':>7} | {'Peak Ret':>10} | {'Final Ret':>10} | "
              f"{'SPS':>8} | {'Time(min)':>10} | {'Speedup':>8}")
        base: Optional[float] = None
        for g in sorted(results[env]):
            r = results[env][g]
            if base is None:
                base = r["time_min"]
            sp = base / r["time_min"] if r["time_min"] > 0 else 0
            print(f"{g:5d} | {g*8:7d} | {r['peak_return']:10.1f} | {r['final_return']:10.1f} | "
                  f"{r['mean_sps']:8.0f} | {r['time_min']:10.1f} | {sp:7.2f}×")

    # Save CSV summary
    out = os.path.join(args.input_dir, "scaling_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "gpus", "actors", "peak_return", "final_return",
                     "mean_sps", "time_min", "total_steps"])
        for env in args.envs:
            for g in sorted(results.get(env, {})):
                r = results[env][g]
                w.writerow([env, g, g*8, f"{r['peak_return']:.1f}",
                           f"{r['final_return']:.1f}", f"{r['mean_sps']:.0f}",
                           f"{r['time_min']:.1f}", r['total_steps']])
    print(f"\n汇总已保存: {out}")


if __name__ == "__main__":
    main()