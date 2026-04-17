#!/usr/bin/env python3
"""
plot_training.py — 单环境详细训练指标图 (Nature 期刊风格)
Per-environment detailed training figure (Nature journal style)

为每个 MuJoCo 环境分别生成一张包含所有关键训练指标的多面板图,
同时展示分布式 vs 单机两种模式的训练曲线。

Usage:
    python scripts/plot_training.py \
        --env hopper \
        --dist_csv   output/metrics.csv \
        --single_csv output/metrics_single.csv \
        --output_dir output/figures

    python scripts/plot_training.py --all   # 一次生成全部 3 个环境
"""

import sys, os, csv, argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 将 NaturePlotter 所在目录加入 sys.path ----
BINDINGSPLOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "nature-bindingsplot", "scripts"
)
if os.path.isdir(BINDINGSPLOT_DIR):
    sys.path.insert(0, BINDINGSPLOT_DIR)
else:
    _alt = "/opt/tiger/mira_nas/plugins/prod/custom/9565078/skills/nature-bindingsplot/scripts"
    if os.path.isdir(_alt):
        sys.path.insert(0, _alt)

try:
    from bindingsplot_nature import NaturePlotter
    _HAS_NATURE_PLOTTER = True
except ImportError:
    _HAS_NATURE_PLOTTER = False
    print("[Warning] nature-bindingsplot not found, using matplotlib fallback.", flush=True)

    class NaturePlotter:
        """matplotlib 回退实现，模拟 Nature 期刊风格"""
        DOUBLE_COL = 7.2

        _NPG_COLORS = [
            "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
            "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
            "#7E6148", "#B09C85",
        ]

        def __init__(self, style="nature", palette="npg", use_latex=False):
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                "font.size": 7,
                "mathtext.fontset": "custom",
                "mathtext.rm": "Helvetica",
                "mathtext.it": "Helvetica:italic",
                "mathtext.bf": "Helvetica:bold",
                "axes.linewidth": 0.5,
                "axes.edgecolor": "#000000",
                "axes.labelsize": 7,
                "axes.titlesize": 7,
                "axes.titleweight": "bold",
                "axes.labelweight": "normal",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.width": 0.3,
                "ytick.minor.width": 0.3,
                "xtick.major.size": 3.0,
                "ytick.major.size": 3.0,
                "xtick.minor.size": 1.5,
                "ytick.minor.size": 1.5,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "legend.fontsize": 5.5,
                "legend.frameon": True,
                "legend.fancybox": False,
                "legend.edgecolor": "#cccccc",
                "legend.borderpad": 0.3,
                "legend.handletextpad": 0.4,
                "lines.linewidth": 1.0,
                "lines.markersize": 3,
                "figure.dpi": 150,
                "figure.facecolor": "white",
                "savefig.dpi": 300,
                "savefig.facecolor": "white",
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                "axes.prop_cycle": plt.cycler(color=self._NPG_COLORS),
            })

        def plot_multi(self, bindingsplot_funcs, nrows, ncols, figsize,
                       sharex=False, sharey=False):
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                     sharex=sharex, sharey=sharey, squeeze=False)
            for idx, func in enumerate(bindingsplot_funcs):
                r, c = divmod(idx, ncols)
                func(axes[r, c])
            return fig, axes

        def save(self, path, dpi=300):
            plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
            plt.close()

# ===================== 工具函数 =====================

def load_csv(path: str) -> dict:
    """读取 CSV, 返回 {列名: np.array} 字典"""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV 为空: {path}")
    data = {}
    for key in rows[0].keys():
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        data[key] = np.array(vals)
    return data


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """滑动平均平滑"""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="same")
    for i in range(window // 2):
        smoothed[i] = np.mean(y[: i + window // 2 + 1])
        smoothed[-(i + 1)] = np.mean(y[-(i + window // 2 + 1):])
    return smoothed


# ===================== 配置 =====================

COLOR_DIST = "#E64B35"       # NPG 红 — Distributed
COLOR_SINGLE = "#3C5488"     # NPG 深蓝 — Single
FILL_ALPHA = 0.12
SMOOTH_W = 7

# 11 个指标 + 1 个 summary = 12 格 → 4×3 布局
DETAIL_METRICS = [
    # (csv_key,         Y_label,                smooth?)
    ("avg_return",      "Average Return",       True),
    ("sps",             "SPS (steps/s)",         True),
    ("loss",            "Total Loss",           True),
    ("policy_loss",     "Policy Loss",          True),
    ("value_loss",      "Value Loss",           True),
    ("entropy",         "Entropy",              True),
    ("approx_kl",       "Approx KL",            True),
    ("clip_fraction",   "Clip Fraction",        True),
    ("explained_var",   "Explained Variance",   True),
    ("grad_norm",       "Gradient Norm",        True),
    ("lr",              "Learning Rate",        False),
]

# 环境预设
ENV_PRESETS = {
    "hopper": {
        "name": "Hopper-v5",
        "dist":   "output/metrics.csv",
        "single": "output/metrics_single.csv",
    },
    "walker2d": {
        "name": "Walker2d-v5",
        "dist":   "output/walker2d/metrics.csv",
        "single": "output/walker2d/metrics_single.csv",
    },
    "halfcheetah": {
        "name": "HalfCheetah-v5",
        "dist":   "output/halfcheetah/metrics.csv",
        "single": "output/halfcheetah/metrics_single.csv",
    },
}


# ===================== 回调工厂 =====================

def make_detail_callback(dist_data, single_data, metric_key, ylabel,
                         do_smooth, show_legend=False):
    """生成单个指标子图回调"""
    def callback(ax):
        x_dist = dist_data["total_steps"] / 1e6
        x_single = single_data["total_steps"] / 1e6
        y_d_raw = dist_data[metric_key]
        y_s_raw = single_data[metric_key]

        if do_smooth:
            y_d = smooth(y_d_raw, SMOOTH_W)
            y_s = smooth(y_s_raw, SMOOTH_W)
        else:
            y_d = y_d_raw
            y_s = y_s_raw

        ax.plot(x_dist, y_d, color=COLOR_DIST, linewidth=1.0,
                label="Distributed (8 Actors)", zorder=3)
        if do_smooth:
            ax.fill_between(x_dist, y_d_raw, y_d, color=COLOR_DIST,
                            alpha=FILL_ALPHA, zorder=2)

        ax.plot(x_single, y_s, color=COLOR_SINGLE, linewidth=1.0,
                label="Single", zorder=3)
        if do_smooth:
            ax.fill_between(x_single, y_s_raw, y_s, color=COLOR_SINGLE,
                            alpha=FILL_ALPHA, zorder=2)

        ax.set_xlabel("Total Steps (×10⁶)", fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=6.5)
        ax.tick_params(axis="both", labelsize=5.5)

        if show_legend:
            ax.legend(fontsize=5, frameon=True, fancybox=False,
                      edgecolor="#cccccc", loc="best")

    return callback


def make_summary_callback(dist_data, single_data, env_name):
    """最后一个子图: 文字汇总面板"""
    def callback(ax):
        ax.axis("off")

        dist_ret = dist_data["avg_return"]
        single_ret = single_data["avg_return"]
        dist_sps = dist_data["sps"]
        single_sps = single_data["sps"]

        lines = [
            f"  {env_name}  Summary",
            f"{'─' * 32}",
            f"  Dist  peak return:  {np.nanmax(dist_ret):.0f}",
            f"  Single peak return: {np.nanmax(single_ret):.0f}",
            f"  Dist  final return: {dist_ret[-1]:.0f}",
            f"  Single final return:{single_ret[-1]:.0f}",
            f"  Mean SPS ratio:     "
            f"{np.nanmean(dist_sps)/np.nanmean(single_sps):.2f}×",
            f"  Dist  mean SPS:     {np.nanmean(dist_sps):.0f}",
            f"  Single mean SPS:    {np.nanmean(single_sps):.0f}",
        ]
        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=6, fontfamily="monospace",
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                          edgecolor="#cccccc", linewidth=0.5))
    return callback


# ===================== 单环境绘图 =====================

def plot_env(env_key: str, dist_csv: str, single_csv: str, output_dir: str):
    """为一个环境绘制详细训练图"""
    preset = ENV_PRESETS.get(env_key, {})
    env_name = preset.get("name", env_key)

    dist_data = load_csv(dist_csv)
    single_data = load_csv(single_csv)

    nrows, ncols = 4, 3
    callbacks = []

    for i, (metric_key, ylabel, do_sm) in enumerate(DETAIL_METRICS):
        show_leg = (i == 0)
        cb = make_detail_callback(dist_data, single_data, metric_key,
                                  ylabel, do_sm, show_legend=show_leg)
        callbacks.append(cb)

    # 最后一格: 汇总面板
    callbacks.append(make_summary_callback(dist_data, single_data, env_name))

    # ---- 绘制 ----
    plotter = NaturePlotter(style="nature", palette="npg", use_latex=False)

    fig, axes = plotter.plot_multi(
        bindingsplot_funcs=callbacks,
        nrows=nrows,
        ncols=ncols,
        figsize=(NaturePlotter.DOUBLE_COL, 8.5),  # 全宽 7.2 × 8.5 in
        sharex=False,
        sharey=False,
    )

    fig.suptitle(
        f"{env_name} — Training Details (Distributed vs Single)",
        fontsize=8, fontweight="bold", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.7, w_pad=0.5)

    # ---- 保存 ----
    os.makedirs(output_dir, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = os.path.join(output_dir, f"{env_key}_detail.{fmt}")
        plotter.save(out_path, dpi=300)
        print(f"Saved: {out_path}")


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(
        description="Per-env detail figure (Nature style)")
    parser.add_argument("--env", type=str, default=None,
                        choices=list(ENV_PRESETS.keys()),
                        help="指定环境; 不指定时需要 --all")
    parser.add_argument("--all", action="store_true",
                        help="一次生成全部 3 个环境的图")
    parser.add_argument("--dist_csv", type=str, default=None)
    parser.add_argument("--single_csv", type=str, default=None)
    parser.add_argument("--output_dir", default="output/figures")
    args = parser.parse_args()

    if args.all:
        for env_key, preset in ENV_PRESETS.items():
            print(f"\n{'='*50}")
            print(f"Generating {preset['name']} ...")
            print(f"{'='*50}")
            plot_env(env_key, preset["dist"], preset["single"],
                     args.output_dir)
    elif args.env:
        preset = ENV_PRESETS[args.env]
        dist_csv = args.dist_csv or preset["dist"]
        single_csv = args.single_csv or preset["single"]
        plot_env(args.env, dist_csv, single_csv, args.output_dir)
    else:
        parser.error("请指定 --env <name> 或 --all")


if __name__ == "__main__":
    main()