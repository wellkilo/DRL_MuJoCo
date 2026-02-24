from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    repo_root = Path(__file__).parent.parent
    metrics_path = repo_root / "output" / "metrics.csv"
    if not metrics_path.exists():
        print("output/metrics.csv not found")
        return

    steps: list[int] = []
    sps: list[float] = []
    avg_return: list[float] = []
    loss: list[float] = []
    buffer_size: list[float] = []
    last_step = -1

    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_val = int(float(row["step"])) if row.get("step") else 0
            if steps and step_val <= last_step:
                steps = []
                sps = []
                avg_return = []
                loss = []
                buffer_size = []
            steps.append(step_val)
            sps.append(float(row["sps"]) if row.get("sps") else 0.0)
            avg_return.append(float(row["avg_return"]) if row.get("avg_return") else 0.0)
            loss.append(float(row["loss"]) if row.get("loss") else 0.0)
            buffer_size.append(float(row["buffer_size"]) if row.get("buffer_size") else 0.0)
            last_step = step_val

    if not steps:
        print("output/metrics.csv is empty")
        return

    output_dir = repo_root / "output"
    output_dir.mkdir(exist_ok=True)

    output_files = [
        output_dir / "training_curves.png",
        output_dir / "training_sps.png",
        output_dir / "training_return.png",
        output_dir / "training_loss.png",
        output_dir / "training_buffer.png",
    ]
    for output in output_files:
        if output.exists():
            output.unlink()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    ax1.plot(steps, sps, color="tab:blue", label="SPS")
    ax1.set_ylabel("steps per second")
    ax1.set_title("Training Speed")
    ax1.legend(loc="best")

    ax2.plot(steps, avg_return, color="tab:orange", label="avg_return")
    ax2.set_ylabel("avg return")
    ax2.set_title("Average Return")
    ax2.legend(loc="best")

    ax3.plot(steps, loss, color="tab:red", label="loss")
    ax3.set_xlabel("step")
    ax3.set_ylabel("loss")
    ax3.set_title("Loss")
    ax3.legend(loc="best")

    ax4.plot(steps, buffer_size, color="tab:green", label="buffer_size")
    ax4.set_xlabel("step")
    ax4.set_ylabel("buffer size")
    ax4.set_title("Replay Buffer Size")
    ax4.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
