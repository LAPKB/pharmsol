"""Plot Figures 1, 2, 5, and 6 from paper estimation CSV outputs.

The estimation example owns simulation, likelihoods, and fitted weights. This
script only reads those numerical outputs and renders empirical/discrete
population distributions.

Usage:
    python paper/plot_population_distributions.py
    python paper/plot_population_distributions.py paper/output
"""

from __future__ import annotations

import csv
import math
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required; install it with `python -m pip install matplotlib`"
    ) from exc


DEFAULT_OUTPUT = Path("paper/output")
OBSERVATION_TIMES = (0.2, 0.4, 0.6, 0.8, 1.0)


def read_population(path: Path) -> list[float]:
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            values = [float(row["ke0"]) for row in reader]
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        raise ValueError(f"cannot read population file {path}: {exc}") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"population file {path} has no finite values")
    return values


def read_distribution(path: Path) -> tuple[list[float], list[float]]:
    supports: list[float] = []
    weights: list[float] = []
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                supports.append(float(row["ke0"]))
                weights.append(float(row["weight"]))
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        raise ValueError(f"cannot read distribution file {path}: {exc}") from exc
    if not supports or len(supports) != len(weights):
        raise ValueError(f"distribution file {path} is empty or malformed")
    if any(not math.isfinite(value) for value in supports + weights) or any(
        weight < 0 for weight in weights
    ):
        raise ValueError(f"distribution file {path} has invalid values")
    total = sum(weights)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"distribution weights in {path} sum to {total}, not 1")
    return supports, weights


def read_sample(path: Path) -> list[float]:
    try:
        with path.open(newline="") as handle:
            values = [float(row["ke0"]) for row in csv.DictReader(handle)]
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        raise ValueError(f"cannot read sample file {path}: {exc}") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"sample file {path} has no finite values")
    return values


def read_directional_derivative(
    path: Path,
) -> tuple[list[float], list[float], list[bool]]:
    supports: list[float] = []
    derivatives: list[float] = []
    active_supports: list[bool] = []
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                supports.append(float(row["ke0"]))
                derivatives.append(float(row["directional_derivative"]))
                active_supports.append(row["is_fml_support"].lower() == "true")
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        raise ValueError(f"cannot read directional derivative file {path}: {exc}") from exc
    if not supports or len(supports) != len(derivatives):
        raise ValueError(f"directional derivative file {path} is empty or malformed")
    if any(not math.isfinite(value) for value in supports + derivatives):
        raise ValueError(f"directional derivative file {path} has non-finite values")
    return supports, derivatives, active_supports


def validate_observations(path: Path, expected_subjects: int = 100) -> None:
    rows: dict[int, list[float]] = defaultdict(list)
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows[int(row["subject"])].append(float(row["time"]))
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        raise ValueError(f"cannot read observations {path}: {exc}") from exc
    if len(rows) != expected_subjects or any(len(times) != 5 for times in rows.values()):
        raise ValueError(f"{path} does not contain {expected_subjects} subjects × 5 observations")
    for times in rows.values():
        if any(not math.isclose(time, expected) for time, expected in zip(times, OBSERVATION_TIMES)):
            raise ValueError(f"{path} has an unexpected observation grid")


def x_limits(values: list[float]) -> tuple[float, float]:
    return min(0.2, min(values) - 0.02), max(2.0, max(values) + 0.02)


def save_figure(fig, output: Path) -> None:
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_figure_1(output_dir: Path, population: list[float]) -> None:
    bins: dict[float, float] = {}
    try:
        with (output_dir / "figure1_initial_k0_distribution.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                bins[float(row["k0_bin"])] = float(row["relative_frequency"])
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        raise ValueError("cannot read Figure 1 bin data") from exc

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.bar(
        list(bins),
        list(bins.values()),
        width=0.01,
        color="steelblue",
        edgecolor="steelblue",
        align="center",
    )
    ax.set_xlabel("Initial elimination rate K(0)")
    ax.set_ylabel("Relative frequency")
    ax.set_xlim(*x_limits(population))
    ax.set_ylim(0, max(0.09, max(bins.values()) * 1.15))
    ax.grid(True, linewidth=0.3, alpha=0.35)
    save_figure(fig, output_dir / "figure1_initial_k0_distribution")


def plot_discrete(
    output_dir: Path,
    supports: list[float],
    weights: list[float],
    stem: str,
    ylabel: str,
    scale: float = 1.0,
    ymax: float | None = None,
) -> None:
    scaled = [weight * scale for weight in weights]
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.vlines(supports, 0, scaled, color="steelblue", linewidth=1.0, alpha=0.85)
    ax.plot(supports, scaled, "o", color="steelblue", markersize=2.0)
    ax.set_xlabel("Population elimination rate K0")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*x_limits(supports))
    if ymax is None:
        ymax = max(scaled) * 1.15
    ax.set_ylim(0, max(ymax, max(scaled) * 1.15))
    ax.grid(True, linewidth=0.3, alpha=0.35)
    save_figure(fig, output_dir / stem)


def plot_sample_histogram(output_dir: Path, sample: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.hist(sample, bins=50, range=x_limits(sample), color="steelblue", edgecolor="white")
    ax.set_xlabel("Population elimination rate K0")
    ax.set_ylabel("Sample count (N=100)")
    ax.set_xlim(*x_limits(sample))
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    save_figure(fig, output_dir / "figure5_estimated_k0_sample_histogram")


def plot_figure_6(output_dir: Path) -> None:
    supports, derivatives, active_supports = read_directional_derivative(
        output_dir / "figure6_directional_derivative.csv"
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(supports, derivatives, color="steelblue", linewidth=1.0)
    active_x = [value for value, active in zip(supports, active_supports) if active]
    active_y = [value for value, active in zip(derivatives, active_supports) if active]
    ax.plot(active_x, active_y, "o", color="steelblue", markersize=2.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Population elimination rate K0")
    ax.set_ylabel("Directional derivative D(K0, FML)")
    ax.set_xlim(min(supports), max(supports))
    ax.grid(True, linewidth=0.3, alpha=0.35)
    save_figure(fig, output_dir / "figure6_directional_derivative")


def ks_statistic(left: list[float], right: list[float]) -> float:
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    values = sorted(set(left_sorted + right_sorted))
    return max(
        abs(
            bisect_right(left_sorted, value) / len(left_sorted)
            - bisect_right(right_sorted, value) / len(right_sorted)
        )
        for value in values
    )


def ks_asymptotic_pvalue(statistic: float, n_left: int, n_right: int) -> float:
    effective_n = n_left * n_right / (n_left + n_right)
    lam = (effective_n**0.5 + 0.12 + 0.11 / effective_n**0.5) * statistic
    return max(
        0.0,
        min(
            1.0,
            2.0
            * sum(
                (-1.0) ** (j - 1) * math.exp(-2.0 * (j * lam) ** 2)
                for j in range(1, 100)
            ),
        ),
    )


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    population = read_population(output_dir / "population_ke0.csv")
    if len(population) != 100:
        raise ValueError(f"expected 100 population values, found {len(population)}")
    validate_observations(output_dir / "experiment1_observations.csv")
    validate_observations(output_dir / "experiment2_observations.csv")

    plot_figure_1(output_dir, population)
    supports_1, weights_1 = read_distribution(output_dir / "figure2_fml_sigma_ke_0.csv")
    supports_2, weights_2 = read_distribution(output_dir / "figure5_fml_sigma_ke_0_5.csv")
    plot_discrete(
        output_dir,
        supports_1,
        weights_1,
        "figure2_estimated_k0_sigma_ke_0",
        "Probability mass",
        ymax=0.08,
    )
    plot_discrete(
        output_dir,
        supports_2,
        weights_2,
        "figure5_estimated_k0_sigma_ke_0_5",
        "Probability mass × 100 (historical scale)",
        scale=100.0,
        ymax=14.0,
    )
    plot_discrete(
        output_dir,
        supports_2,
        weights_2,
        "figure5_estimated_k0_probability",
        "Probability mass",
    )

    sample_1 = read_sample(output_dir / "figure2_fml_sample_n100.csv")
    sample_2 = read_sample(output_dir / "figure5_fml_sample_n100.csv")
    plot_sample_histogram(output_dir, sample_2)
    plot_figure_6(output_dir)
    for label, sample in [("Experiment 1", sample_1), ("Experiment 2", sample_2)]:
        statistic = ks_statistic(population, sample)
        pvalue = ks_asymptotic_pvalue(statistic, len(population), len(sample))
        print(f"{label} KS statistic={statistic:.6f}, asymptotic p-value={pvalue:.6f}")
    print(f"wrote Figures 1, 2, 5, and 6 under {output_dir}")


if __name__ == "__main__":
    main()
