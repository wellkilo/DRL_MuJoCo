#!/usr/bin/env python3
"""
plot_comparison.py — 多环境总览对比图 (Nature 期刊风格)
Multi-environment overview comparison figure (Nature journal style)

使用 NaturePlotter (nature-bindingsplot skill) 绘制 3 环境 × 多指标 的大型组图，
分布式 (8 Actors) vs 单机模式 对比。

Usage:
    python scripts/plot_comparison.py \
        --output_dir output/figures \
        --hopper_dist   output/metrics.csv \
        --hopper_single output/metrics_single.csv \
        --walker_dist   output/walker2d/metrics.csv \
        --walker_single output/walker2d/metrics_single.csv \
        --cheetah_dist  output/halfcheetah/metrics.csv \
        --cheetah_single output/halfcheetah/metrics_single.csv
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
    print("[Warning] Install with: pip install nature-bindingsplot", flush=True)

    class NaturePlotter:
        """matplotlib 回退实现，模拟 Nature 期刊风格"""
        DOUBLE_COL = 7.2

        # NPG (Nature Publishing Group) 配色方案
        _NPG_COLORS = [
            "#E64B35",  # 红
            "#4DBBD5",  # 青
            "#00A087",  # 绿
            "#3C5488",  # 深蓝
            "#F39B7F",  # 珊瑚
            "#8491B4",  # 灰蓝
            "#91D1C2",  # 薄荷
            "#DC0000",  # 纯红
            "#7E6148",  # 棕
            "#B09C85",  # 卡其
        ]

        def __init__(self, style="nature", palette="npg", use_latex=False):
            """初始化 Nature 期刊风格 rcParams"""
            plt.rcParams.update({
                # 字体：Nature 使用 sans-serif (Helvetica/Arial)
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                "font.size": 7,
                "mathtext.fontset": "custom",
                "mathtext.rm": "Helvetica",
                "mathtext.it": "Helvetica:italic",
                "mathtext.bf": "Helvetica:bold",
                # 坐标轴：细线，Nature 风格
                "axes.linewidth": 0.5,
                "axes.edgecolor": "#000000",
                "axes.labelsize": 7,
                "axes.titlesize": 7,
                "axes.titleweight": "bold",
                "axes.labelweight": "normal",
                "axes.spines.top": False,
                "axes.spines.right": False,
                # 刻度
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
                # 图例
                "legend.fontsize": 5.5,
                "legend.frameon": True,
                "legend.fancybox": False,
                "legend.edgecolor": "#cccccc",
                "legend.borderpad": 0.3,
                "legend.handletextpad": 0.4,
                # 线条
                "lines.linewidth": 1.0,
                "lines.markersize": 3,
                # 图形
                "figure.dpi": 150,
                "figure.facecolor": "white",
                "savefig.dpi": 300,
                "savefig.facecolor": "white",
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                # 颜色循环
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
    """简单滑动平均平滑"""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="same")
    for i in range(window // 2):
        smoothed[i] = np.mean(y[: i + window // 2 + 1])
        smoothed[-(i + 1)] = np.mean(y[-(i + window // 2 + 1):])
    return smoothed


# ===================== 绘图配置 =====================

COLOR_DIST = "#E64B35"       # NPG 红 — Distributed
COLOR_SINGLE = "#3C5488"     # NPG 深蓝 — Single
FILL_ALPHA = 0.15
SMOOTH_WINDOW = 7

# 总览图展示的 6 个关键指标
OVERVIEW_METRICS = [
    ("avg_return",     "Average Return"),
    ("sps",            "SPS (steps/s)"),
    ("entropy",        "Policy Entropy"),
    ("approx_kl",      "Approx KL"),
    ("clip_fraction",  "Clip Fraction"),
    ("explained_var",  "Explained Variance"),
]

ENV_NAMES = ["Hopper-v5", "Walker2d-v5", "HalfCheetah-v5"]


# ===================== 回调工厂 =====================

def make_subplot_callback(dist_data, single_data, metric_key, ylabel,
                          env_name, show_legend=False):
    """
    返回 func(ax) -> None 回调, 用于 NaturePlotter.plot_multi.
    在 ax 上绘制 distributed vs single 的对比曲线.
    """
    def callback(ax):
        x_dist = dist_data["total_steps"] / 1e6
        x_single = single_data["total_steps"] / 1e6
        y_dist_raw = dist_data[metric_key]
        y_single_raw = single_data[metric_key]

        y_dist = smooth(y_dist_raw, SMOOTH_WINDOW)
        y_single = smooth(y_single_raw, SMOOTH_WINDOW)

        # 分布式曲线 (红色)
        ax.plot(x_dist, y_dist, color=COLOR_DIST, linewidth=1.0,
                label="Distributed (8 Actors)", zorder=3)
        ax.fill_between(x_dist, y_dist_raw, y_dist, color=COLOR_DIST,
                        alpha=FILL_ALPHA, zorder=2)

        # 单机曲线 (深蓝)
        ax.plot(x_single, y_single, color=COLOR_SINGLE, linewidth=1.0,
                label="Single", zorder=3)
        ax.fill_between(x_single, y_single_raw, y_single, color=COLOR_SINGLE,
                        alpha=FILL_ALPHA, zorder=2)

        ax.set_xlabel("Total Steps (×10⁶)", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(env_name, fontsize=7, fontweight="bold", pad=4)
        ax.tick_params(axis="both", labelsize=6)

        if show_legend:
            ax.legend(fontsize=5.5, frameon=True, fancybox=False,
                      edgecolor="#cccccc", loc="best")

    return callback


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-env overview comparison (Nature style)")
    parser.add_argument("--hopper_dist",   default="output/metrics.csv")
    parser.add_argument("--hopper_single", default="output/metrics_single.csv")
    parser.add_argument("--walker_dist",   default="output/walker2d/metrics.csv")
    parser.add_argument("--walker_single", default="output/walker2d/metrics_single.csv")
    parser.add_argument("--cheetah_dist",  default="output/halfcheetah/metrics.csv")
    parser.add_argument("--cheetah_single",default="output/halfcheetah/metrics_single.csv")
    parser.add_argument("--output_dir",    default="output/figures")
    args = parser.parse_args()

    # ---- 加载数据 ----
    env_data = [
        (load_csv(args.hopper_dist),  load_csv(args.hopper_single)),
        (load_csv(args.walker_dist),  load_csv(args.walker_single)),
        (load_csv(args.cheetah_dist), load_csv(args.cheetah_single)),
    ]

    # ---- 构建回调: 6行 × 3列 ----
    nrows = len(OVERVIEW_METRICS)   # 6
    ncols = len(ENV_NAMES)           # 3
    callbacks = []
    for row_idx, (metric_key, ylabel) in enumerate(OVERVIEW_METRICS):
        for col_idx, env_name in enumerate(ENV_NAMES):
            dist_d, single_d = env_data[col_idx]
            show_leg = (row_idx == 0 and col_idx == 0)
            cb = make_subplot_callback(
                dist_d, single_d, metric_key, ylabel,
                env_name if row_idx == 0 else "",
                show_legend=show_leg
            )
            callbacks.append(cb)

    # ---- 绘制 ----
    plotter = NaturePlotter(style="nature", palette="npg", use_latex=False)

    fig, axes = plotter.plot_multi(
        bindingsplot_funcs=callbacks,
        nrows=nrows,
        ncols=ncols,
        figsize=(NaturePlotter.DOUBLE_COL, 9.5),  # 7.2 × 9.5 in
        sharex=False,
        sharey=False,
    )

    # 仅最末行显示 X 轴标签
    for idx, ax in enumerate(axes.flat):
        row = idx // ncols
        if row < nrows - 1:
            ax.set_xlabel("")

    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=0.8, w_pad=0.6)

    # ---- 保存 ----
    os.makedirs(args.output_dir, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = os.path.join(args.output_dir, f"overview_comparison.{fmt}")
        plotter.save(out_path, dpi=300)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()