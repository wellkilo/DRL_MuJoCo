from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    sps: list[float] = []
    avg_return: list[float] = []
    last_step = -1

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
            steps.append(step_val)
            sps.append(float(row["sps"]) if row.get("sps") else 0.0)
            avg_return.append(float(row["avg_return"]) if row.get("avg_return") else 0.0)
            last_step = step_val

    return steps, sps, avg_return


def main() -> None:
    repo_root = Path(__file__).parent.parent
    dist_metrics = repo_root / "output" / "metrics.csv"
    single_metrics = repo_root / "output" / "metrics_single.csv"

    dist_steps, dist_sps, dist_return = load_metrics(dist_metrics)
    single_steps, single_sps, single_return = load_metrics(single_metrics)

    output_dir = repo_root / "output"
    output_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if dist_steps and dist_sps:
        ax1.plot(dist_steps, dist_sps, color="tab:blue", label="Distributed (8 actors)")
    if single_steps and single_sps:
        ax1.plot(single_steps, single_sps, color="tab:orange", label="Single (1 actor)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("steps per second")
    ax1.set_title("Training Speed (SPS)")
    ax1.legend(loc="best")

    if dist_steps and dist_return:
        ax2.plot(dist_steps, dist_return, color="tab:blue", label="Distributed (8 actors)")
    if single_steps and single_return:
        ax2.plot(single_steps, single_return, color="tab:orange", label="Single (1 actor)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("avg return")
    ax2.set_title("Average Return")
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
