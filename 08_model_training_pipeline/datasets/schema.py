"""Canonical data schemas for trajectory-prediction training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GoalFeatures:
    """Goal context expressed relative to the current ego state."""

    goal_x_world: float
    goal_y_world: float
    rel_goal_x_ego: float
    rel_goal_y_ego: float
    goal_distance: float
    goal_heading_error: float


@dataclass
class EgoStateStep:
    """One past ego step used for temporal conditioning."""

    x_world: float
    y_world: float
    yaw_world: float
    linear_x: float
    angular_z: float


@dataclass
class ScanFeatures:
    """Optional scan tensor metadata."""

    path: str | None
    num_beams: int
    channels: int
    encoding: str = "range_intensity"


@dataclass
class GraphFeatures:
    """Optional multi-agent graph metadata."""

    node_order: list[str]
    node_dim: int
    edge_dim: int
    path: str | None = None


@dataclass
class TargetFuture:
    """Primary supervised target: local future trajectory."""

    future_xy_local: list[list[float]]
    future_dt: list[float]


@dataclass
class SampleMetadata:
    """Everything needed to trace a sample back to its source."""

    episode_id: str
    sample_id: str
    timestamp_ns: int
    source_kind: str
    source_path: str
    ego_id: str = "husky_local"
    teacher_state: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectorySample:
    """Canonical train-ready sample shared by all models."""

    past_len: int
    future_len: int
    ego_past: list[EgoStateStep]
    goal: GoalFeatures
    target: TargetFuture
    scan: ScanFeatures | None = None
    graph: GraphFeatures | None = None
    metadata: SampleMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_sample_schema() -> dict[str, Any]:
    """Machine-readable schema summary for exporters and validators."""

    return {
        "task": "goal_reaching_trajectory_prediction",
        "target": {
            "name": "future_xy_local",
            "shape": ["future_len", 2],
            "description": "Future ego trajectory in the ego-local frame as (dx, dy).",
        },
        "required_groups": [
            "ego_past",
            "goal",
            "target",
        ],
        "optional_groups": [
            "scan",
            "graph",
            "metadata",
        ],
        "ego_step_fields": [
            "x_world",
            "y_world",
            "yaw_world",
            "linear_x",
            "angular_z",
        ],
        "goal_fields": [
            "goal_x_world",
            "goal_y_world",
            "rel_goal_x_ego",
            "rel_goal_y_ego",
            "goal_distance",
            "goal_heading_error",
        ],
    }

