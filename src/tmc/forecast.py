"""Short-horizon forecasts from detector-based estimates.

    python -m tmc.forecast        # small example on a synthetic step in demand

  * ExpSmoothing: simple exponential smoothing of a measured rate (e.g. ramp demand from R_dem),
    level L_k = alpha x_k + (1 - alpha) L_{k-1}; the h-step forecast is the level (no trend).
    Uncertainty = RMS of the one-step-ahead errors seen so far (None until two errors exist).
  * Persistence (forecast = last measurement) is the naive baseline the smoother must beat
    (evaluated in tmc.evaluate).
  * queue_forecast: point-queue mass balance over the horizon, N(t+h) = max(0, N + (lambda - r) h).
Sources in docs/source_to_design.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ExpSmoothing:
    alpha: float = 0.3
    level: float | None = None
    _sq_err: list = field(default_factory=list)

    def update(self, x: float | None) -> float | None:
        """Add one measurement (None = missing: keep the level). Returns the new level."""
        if x is None:
            return self.level
        if self.level is None:
            self.level = float(x)
        else:
            self._sq_err.append((x - self.level) ** 2)   # one-step-ahead error of the previous forecast
            self.level = self.alpha * x + (1 - self.alpha) * self.level
        return self.level

    def forecast(self) -> float | None:
        return self.level

    @property
    def rmse(self) -> float | None:
        if len(self._sq_err) < 2:
            return None
        return math.sqrt(sum(self._sq_err) / len(self._sq_err))


def queue_forecast(n_now: float | None, arrival_vph: float | None, release_vph: float, horizon_s: float) -> float | None:
    """Point-queue mass balance. None if either input is missing."""
    if n_now is None or arrival_vph is None:
        return None
    return max(0.0, n_now + (arrival_vph - release_vph) * horizon_s / 3600.0)


def main() -> None:
    es = ExpSmoothing(alpha=0.3)
    xs = [300] * 6 + [500] * 10
    print("k  meas  level")
    for k, x in enumerate(xs):
        print(f"{k:2d} {x:5d} {es.update(x):6.1f}")
    print("rmse of one-step errors:", round(es.rmse, 1))
    print("queue in 60 s from N=20, lambda=500, r=300:", queue_forecast(20, 500, 300, 60))


if __name__ == "__main__":
    main()
