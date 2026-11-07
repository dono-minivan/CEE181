"""Minimum exchange formats from the CEE 181 design guide.

Units: seconds and meters internally (convert only at documented boundaries).
Unavailable quantities are None, never silently 0.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VehicleState:
    time_s: float
    vehicle_id: int
    lane_id: int
    position_m: float
    speed_mps: float
    acceleration_mps2: Optional[float] = None


@dataclass(frozen=True)
class Observation:
    sensor_id: str
    interval_start_s: float
    interval_end_s: float
    count: int
    mean_speed_mps: Optional[float]  # None when not measurable
    quality_flag: str = "ok"         # e.g. ok, partial, missing, delayed, noisy
    occupancy: Optional[float] = None  # fraction of the interval the detector was occupied; None if not a loop


@dataclass(frozen=True)
class EstimateForecast:
    issue_time_s: float
    target_time_s: float
    segment_id: str
    variable_name: str
    value: float
    units: str
    uncertainty: Optional[float] = None


@dataclass(frozen=True)
class ControlCommand:
    issue_time_s: float
    apply_time_s: float
    actuator_id: str
    action: str
    value: float
    units: str
    min_value: float
    max_value: float
    fallback: str


@dataclass(frozen=True)
class EvaluationRecord:
    scenario_id: str
    demand_id: str
    model_version: str
    controller_version: str
    metrics: dict
    seed: Optional[int] = None
