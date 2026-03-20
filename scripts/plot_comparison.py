from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path, sample_interval: int = 1) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    sps: list[float] = []
    avg_return: list[float] = []
    last_step = -1
    count = 0

    if not metrics_path.exists():
        return steps, sps, avg_return

    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_val = int(float(row["step"])) if row.get("step") else 0
            if steps and step_val <= last_step:
                steps = []
                sps = []
                avg_return = []
                count = 0
            
            # 按间隔采样
            if count % sample_interval == 0:
                steps.append(step_val)
                sps.append(float(row["sps"]) if row.get("sps") else 0.0)
                avg_return.append(float(row["avg_return"]) if row.get("avg_return") else 0.0)
            
            last_step = step_val
            count += 1

    return steps, sps, avg_return


def main() -> None:
    repo_root = Path(__file__).parent.parent
    dist_metrics = repo_root / "output" / "metrics.csv"
    single_metrics = repo_root / "output" / "metrics_single.csv"

    # 不再需要采样间隔，因为现在两个配置的训练步数计数方式一致了
    dist_steps, dist_sps, dist_return = load_metrics(dist_metrics, sample_interval=1)
    single_steps, single_sps, single_return = load_metrics(single_metrics, sample_interval=1)

    output_dir = repo_root / "output"
    output_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if dist_steps and dist_sps:
        ax1.plot(dist_steps, dist_sps, color="tab:blue", label="Distributed (8 actors)", marker='o', markersize=4, linestyle='-')
    if single_steps and single_sps:
        ax1.plot(single_steps, single_sps, color="tab:orange", label="Single (1 actor)", marker='s', markersize=4, linestyle='-')
    ax1.set_xlabel("step")
    ax1.set_ylabel("steps per second")
    ax1.set_title("Training Speed (SPS)")
    ax1.legend(loc="best")

    if dist_steps and dist_return:
        ax2.plot(dist_steps, dist_return, color="tab:blue", label="Distributed (8 actors)", marker='o', markersize=4, linestyle='-')
    if single_steps and single_return:
        ax2.plot(single_steps, single_return, color="tab:orange", label="Single (1 actor)", marker='s', markersize=4, linestyle='-')
    ax2.set_xlabel("step")
    ax2.set_ylabel("avg return")
    ax2.set_title("Average Return")
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
