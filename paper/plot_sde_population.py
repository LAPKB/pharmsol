"""Plot the two paper SDE trajectory figures from the authoritative CSV.

Usage:
    python paper/plot_sde_population.py
    python paper/plot_sde_population.py paper/output/sde_population_trajectories.csv paper/output
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required; install it with `python -m pip install matplotlib`"
    ) from exc


DEFAULT_CSV = Path("paper/output/sde_population_trajectories.csv")
DEFAULT_OUTPUT = Path("paper/output")


def read_trajectories(path: Path):
    rows = defaultdict(lambda: {"step": [], "concentration": [], "ke": []})
    try:
        handle = path.open(newline="")
    except OSError as exc:
        raise ValueError(f"cannot open trajectory CSV {path}: {exc}") from exc

    with handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                subject = int(row["subject"])
                step = int(row["step"])
                concentration = float(row["concentration"])
                ke = float(row["ke"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid trajectory CSV row {row_number} in {path}: {exc}"
                ) from exc
            rows[subject]["step"].append(step)
            rows[subject]["concentration"].append(concentration)
            rows[subject]["ke"].append(ke)
    return [rows[subject] for subject in sorted(rows)]


def plot_trajectories(trajectories, key: str, ylabel: str, output: Path, ylim=None):
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for trajectory in trajectories:
        ax.plot(
            trajectory["step"],
            trajectory[key],
            color="black",
            linewidth=0.45,
            alpha=0.55,
        )
    ax.set_xlabel("Integration step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, trajectories[0]["step"][-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = read_trajectories(csv_path)
    if len(trajectories) != 100:
        raise ValueError(f"expected 100 subjects, found {len(trajectories)}")
    if any(len(trajectory["step"]) != 5001 for trajectory in trajectories):
        raise ValueError("expected 5001 points per subject")
    if any(value != 20.0 for trajectory in trajectories for value in trajectory["concentration"][:1]):
        raise ValueError("all concentration trajectories must start at 20")
    if any(value < 0.0 for trajectory in trajectories for value in trajectory["ke"]):
        raise ValueError("Ke trajectories must be nonnegative")

    plot_trajectories(
        trajectories,
        "concentration",
        "Concentration (X / V)",
        output_dir / "figure3_concentrations",
        ylim=(0, 20),
    )
    plot_trajectories(
        trajectories,
        "ke",
        "Elimination rate Ke",
        output_dir / "figure4_ke_trajectories",
    )
    print(f"wrote figures for {len(trajectories)} subjects from {csv_path}")


if __name__ == "__main__":
    main()
