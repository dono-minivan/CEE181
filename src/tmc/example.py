"""Reproducible example run: python -m tmc.example [path/to/scenario.yaml]

Runs the single-lane baseline, prints a summary (SI and US units), and writes labeled outputs to results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tmc.idm import capacity
from tmc.network import run_scenario

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = ROOT / "configs" / "scenario_s0_single_lane.yaml"
MPS_TO_MPH = 2.236936
M_TO_MI = 1 / 1609.344


def main(scenario: Path = DEFAULT_SCENARIO) -> None:
    res = run_scenario(scenario)
    sid = res.config.scenario_id
    c = capacity(res.params)
    s = res.summary()

    print(f"scenario {sid} | demand {res.config.demand_id} | params {res.config.param_set} | seed {res.config.seed}")
    print(f"IDM capacity (computed): {c['q_c_veh_per_h']:.0f} veh/h at {c['v_c_mps'] * MPS_TO_MPH:.1f} mph")
    for key, val in s.items():
        print(f"  {key}: {val}")
    print(f"  min_speed_mph: {s['min_speed_mps'] * MPS_TO_MPH:.1f}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tr = res.trajectories

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(tr["time_s"] / 60, tr["position_m"] * M_TO_MI, c=tr["speed_mps"] * MPS_TO_MPH,
                    s=0.3, cmap="RdYlGn", vmin=0, vmax=res.params.v0 * MPS_TO_MPH, rasterized=True)
    fig.colorbar(sc, ax=ax, label="speed (mph)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("position from entrance (mi)")
    ax.set_title(f"{sid}: time-space trajectories ({res.config.demand_id}, dt={res.config.dt_s} s)")
    fig.tight_layout()
    fig.savefig(out / f"{sid}_time_space.png", dpi=150)
    plt.close(fig)

    cnt = res.counts
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cnt["time_s"] / 60, cnt["n_arrived"], label="cumulative arrivals")
    ax.plot(cnt["time_s"] / 60, cnt["n_exited"], label="cumulative exits")
    ax.plot(cnt["time_s"] / 60, cnt["n_queued"], label="entry queue")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("vehicles (veh)")
    ax.set_title(f"{sid}: cumulative counts and entry queue ({res.config.demand_id})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{sid}_counts.png", dpi=150)
    plt.close(fig)

    res.vehicles.to_csv(out / f"{sid}_vehicles.csv", index=False)
    print(f"wrote results/{sid}_time_space.png, {sid}_counts.png, {sid}_vehicles.csv")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCENARIO)
