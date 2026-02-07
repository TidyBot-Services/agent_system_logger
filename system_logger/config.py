"""Configuration for the system logger.

This module contains all configuration options for state recording and rewind.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoggerConfig:
    """Configuration for the SystemLogger.

    Attributes:
        max_waypoints: Maximum number of waypoints to store (FIFO buffer).
        record_interval: Time between recordings in seconds.
        base_position_threshold: Minimum base position change to record (meters).
        base_orientation_threshold: Minimum base orientation change to record (radians).
        arm_threshold: Minimum arm joint change to record (radians).
        record_base: Whether to record base state.
        record_arm: Whether to record arm state.
        record_gripper: Whether to record gripper state.
        record_cameras: Whether to record camera frames (not implemented yet).
    """

    # Buffer settings
    max_waypoints: int = 10000

    # Recording rate
    record_interval: float = 0.1  # 10 Hz

    # Movement thresholds for recording (skip if below all thresholds)
    base_position_threshold: float = 0.05    # 5 cm
    base_orientation_threshold: float = 0.1  # ~5.7 degrees
    arm_threshold: float = 0.05              # ~3 degrees

    # What to record
    record_base: bool = True
    record_arm: bool = True
    record_gripper: bool = True
    record_cameras: bool = False  # Not implemented yet

    # Auto-save settings
    auto_save: bool = False
    auto_save_path: Optional[str] = None
    auto_save_interval: float = 60.0  # seconds


@dataclass
class RewindConfig:
    """Configuration for the RewindOrchestrator.

    Attributes:
        settle_time: Time to wait between chunks during rewind (seconds).
        command_rate: Rate to send commands during rewind (Hz).
        chunk_size: Number of waypoints per chunk for smooth interpolation.
        chunk_duration: Duration to execute each chunk (seconds).
        rewind_base: Whether to rewind base.
        rewind_arm: Whether to rewind arm.
        rewind_gripper: Whether to rewind gripper.
        arm_velocity_scale: Scale factor for arm velocity during rewind (0-1).
        base_velocity_scale: Scale factor for base velocity during rewind (0-1).
        safety_margin: Stop rewind this far inside workspace boundary (meters).
    """

    # Timing
    settle_time: float = 0.0       # Time between chunks (reduced from 0.5)
    command_rate: float = 50.0     # Hz - must be > 10 Hz for arm (100ms timeout)

    # Chunked smooth rewind
    chunk_size: int = 30            # Waypoints per chunk (tune for smoothness vs responsiveness)
    chunk_duration: float = 3.0    # Seconds to execute each chunk (arm interpolation time)

    # What to rewind
    rewind_base: bool = True
    rewind_arm: bool = True
    rewind_gripper: bool = False   # Usually don't want to auto-rewind gripper

    # Velocity scaling (slower = safer)
    arm_velocity_scale: float = 0.3
    base_velocity_scale: float = 0.5

    # Safety
    safety_margin: float = 0.1     # meters inside workspace boundary

    # Auto-rewind settings
    auto_rewind_enabled: bool = False
    auto_rewind_percentage: float = 10.0    # % of trajectory to rewind
    monitor_interval: float = 0.1           # seconds between boundary checks

    # Collision detection (active when auto_rewind_enabled is True)
    collision_velocity_threshold: float = 0.3   # actual/commanded ratio below this = collision
    collision_min_cmd_speed: float = 0.05       # m/s minimum commanded speed to consider
    collision_grace_period: float = 0.5         # seconds before triggering


@dataclass
class WorkspaceBounds:
    """Workspace boundary definition for safety checks.

    Attributes:
        base_x_min: Minimum x position for base (meters).
        base_x_max: Maximum x position for base (meters).
        base_y_min: Minimum y position for base (meters).
        base_y_max: Maximum y position for base (meters).
        arm_q_min: Minimum joint angles (radians).
        arm_q_max: Maximum joint angles (radians).
    """

    # Base workspace (rectangle)
    base_x_min: float = -2.0
    base_x_max: float = 2.0
    base_y_min: float = -2.0
    base_y_max: float = 2.0

    # Arm joint limits (Franka Panda defaults)
    arm_q_min: List[float] = field(default_factory=lambda: [
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    ])
    arm_q_max: List[float] = field(default_factory=lambda: [
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
    ])

    def is_base_in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if base position is within bounds.

        Args:
            x: Base x position.
            y: Base y position.
            margin: Safety margin (shrinks bounds).

        Returns:
            True if within bounds.
        """
        return (
            self.base_x_min + margin <= x <= self.base_x_max - margin and
            self.base_y_min + margin <= y <= self.base_y_max - margin
        )

    def is_arm_in_bounds(self, q: List[float], margin: float = 0.0) -> bool:
        """Check if arm joints are within bounds.

        Args:
            q: Joint angles (7 values).
            margin: Safety margin (shrinks bounds).

        Returns:
            True if within bounds.
        """
        if len(q) != 7:
            return True  # Can't check, assume OK

        for i, (qi, qmin, qmax) in enumerate(zip(q, self.arm_q_min, self.arm_q_max)):
            if not (qmin + margin <= qi <= qmax - margin):
                return False
        return True

    def base_distance_to_boundary(self, x: float, y: float) -> dict:
        """Get distances from base position to each boundary edge.

        Args:
            x: Base x position.
            y: Base y position.

        Returns:
            Dict with distances to each edge.
        """
        return {
            "x_min": x - self.base_x_min,
            "x_max": self.base_x_max - x,
            "y_min": y - self.base_y_min,
            "y_max": self.base_y_max - y,
            "min_distance": min(
                x - self.base_x_min,
                self.base_x_max - x,
                y - self.base_y_min,
                self.base_y_max - y,
            ),
        }
