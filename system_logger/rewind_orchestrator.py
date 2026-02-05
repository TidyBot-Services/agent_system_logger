"""Rewind Orchestrator - Coordinated rewind across multiple robot subsystems.

This module provides the RewindOrchestrator class that coordinates rewind
operations across base, arm, and gripper servers.

The orchestrator:
1. Reads waypoints from the SystemLogger
2. Sends coordinated commands to each backend
3. Handles timing and synchronization
4. Provides dry-run capability for testing

Example:
    orchestrator = RewindOrchestrator(logger, config)
    orchestrator.set_backends(base_backend, arm_backend, gripper_backend)

    # Rewind 10% of trajectory
    result = await orchestrator.rewind_percentage(10.0)

    # Rewind to specific waypoint
    result = await orchestrator.rewind_to_waypoint(idx=50)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol

from system_logger.waypoint import UnifiedWaypoint
from system_logger.logger import SystemLogger
from system_logger.config import RewindConfig, WorkspaceBounds

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Log buffer for dashboard display
# -----------------------------------------------------------------------------

class RewindLogBuffer(logging.Handler):
    """Captures rewind-related logs for dashboard display."""

    def __init__(self, max_entries: int = 100):
        super().__init__()
        self.max_entries = max_entries
        self._buffer: deque = deque(maxlen=max_entries)
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record: logging.LogRecord) -> None:
        """Capture log record to buffer."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": self.format(record),
            }
            self._buffer.append(entry)
        except Exception:
            pass  # Don't let logging errors break things

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent log entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of log entries, oldest first (new stuff at bottom).
        """
        entries = list(self._buffer)
        return entries[-limit:]  # Chronological order (oldest first)

    def clear(self) -> None:
        """Clear the log buffer."""
        self._buffer.clear()


# Global log buffer instance
_rewind_log_buffer = RewindLogBuffer()
logger.addHandler(_rewind_log_buffer)


def get_rewind_log_buffer() -> RewindLogBuffer:
    """Get the global rewind log buffer."""
    return _rewind_log_buffer


# -----------------------------------------------------------------------------
# Backend protocols (interfaces)
# -----------------------------------------------------------------------------

class BaseBackendProtocol(Protocol):
    """Protocol for base backend interface."""

    def execute_action(self, x: float, y: float, theta: float) -> None:
        """Move base to position."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get current base state."""
        ...


class ArmBackendProtocol(Protocol):
    """Protocol for arm backend interface."""

    def send_joint_position(self, q: List[float], blocking: bool = True) -> bool:
        """Move arm to joint positions."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get current arm state."""
        ...

    def set_control_mode(self, mode: int) -> bool:
        """Set arm control mode (1 = JOINT_POSITION)."""
        ...


class GripperBackendProtocol(Protocol):
    """Protocol for gripper backend interface."""

    def move(self, position: int, speed: int = 255, force: int = 255) -> tuple:
        """Move gripper to position (0-255)."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get current gripper state."""
        ...


# -----------------------------------------------------------------------------
# Rewind result
# -----------------------------------------------------------------------------

@dataclass
class RewindResult:
    """Result of a rewind operation."""

    success: bool
    steps_rewound: int = 0
    start_waypoint_idx: int = 0
    end_waypoint_idx: int = 0
    error: str = ""
    waypoints_executed: List[Dict[str, Any]] = field(default_factory=list)
    components_rewound: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Rewind Orchestrator
# -----------------------------------------------------------------------------

class RewindOrchestrator:
    """Coordinates rewind operations across all robot subsystems.

    The orchestrator manages the rewind process:
    1. Determines which waypoints to traverse
    2. Sends commands to each backend in sequence
    3. Waits for settling between waypoints
    4. Truncates the trajectory after successful rewind

    Supports selective component rewind (base only, arm only, or both).
    """

    def __init__(
        self,
        system_logger: SystemLogger,
        config: Optional[RewindConfig] = None,
        workspace_bounds: Optional[WorkspaceBounds] = None,
    ):
        """Initialize the rewind orchestrator.

        Args:
            system_logger: SystemLogger instance with recorded waypoints.
            config: Rewind configuration.
            workspace_bounds: Workspace boundary definitions.
        """
        self._logger = system_logger
        self._config = config or RewindConfig()
        self._bounds = workspace_bounds or WorkspaceBounds()

        # Backends (set via set_backends)
        self._base_backend: Optional[BaseBackendProtocol] = None
        self._arm_backend: Optional[ArmBackendProtocol] = None
        self._gripper_backend: Optional[GripperBackendProtocol] = None

        # State
        self._is_rewinding = False

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    @property
    def config(self) -> RewindConfig:
        """Return the rewind configuration."""
        return self._config

    @property
    def is_rewinding(self) -> bool:
        """Return True if a rewind operation is in progress."""
        return self._is_rewinding

    @property
    def trajectory_length(self) -> int:
        """Return the number of waypoints in the trajectory."""
        return len(self._logger)

    def set_backends(
        self,
        base_backend: Optional[BaseBackendProtocol] = None,
        arm_backend: Optional[ArmBackendProtocol] = None,
        gripper_backend: Optional[GripperBackendProtocol] = None,
    ) -> None:
        """Set the backend interfaces for rewind commands.

        Args:
            base_backend: Base control backend.
            arm_backend: Arm control backend.
            gripper_backend: Gripper control backend.
        """
        self._base_backend = base_backend
        self._arm_backend = arm_backend
        self._gripper_backend = gripper_backend

    # -------------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------------

    def is_base_out_of_bounds(self, state: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the base is currently outside the workspace.

        Args:
            state: Current robot state. If None, queries base backend.

        Returns:
            True if base is out of bounds.
        """
        if state is None and self._base_backend:
            state = self._base_backend.get_state()

        if state is None:
            return False

        pose = state.get("base_pose", [0, 0, 0])
        x, y = pose[0], pose[1]

        return not self._bounds.is_base_in_bounds(x, y)

    def get_boundary_status(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed status of current position relative to workspace boundary.

        Args:
            state: Current robot state. If None, queries base backend.

        Returns:
            Boundary status information.
        """
        if state is None and self._base_backend:
            state = self._base_backend.get_state()

        if state is None:
            return {"error": "No state available"}

        pose = state.get("base_pose", [0, 0, 0])
        x, y = pose[0], pose[1]

        distances = self._bounds.base_distance_to_boundary(x, y)

        return {
            "x": x,
            "y": y,
            "out_of_bounds": not self._bounds.is_base_in_bounds(x, y),
            "bounds": {
                "x_min": self._bounds.base_x_min,
                "x_max": self._bounds.base_x_max,
                "y_min": self._bounds.base_y_min,
                "y_max": self._bounds.base_y_max,
            },
            "distances": distances,
        }

    def find_last_safe_waypoint(self) -> Optional[int]:
        """Find the index of the last waypoint where base was within bounds.

        Returns:
            Index of last safe waypoint, or None if none found.
        """
        margin = self._config.safety_margin

        for i in range(len(self._logger) - 1, -1, -1):
            wp = self._logger[i]
            if self._bounds.is_base_in_bounds(wp.x, wp.y, margin=margin):
                return i

        return None

    # -------------------------------------------------------------------------
    # Rewind operations
    # -------------------------------------------------------------------------

    async def rewind_steps(
        self,
        steps: int,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Rewind the robot by a specified number of steps backward.

        Args:
            steps: Number of waypoints to rewind.
            dry_run: If True, only return what would happen.
            components: List of components to rewind ("base", "arm", "gripper").
                        None = use config defaults.

        Returns:
            RewindResult with operation details.
        """
        if self._is_rewinding:
            return RewindResult(success=False, error="Rewind already in progress")

        trajectory_len = len(self._logger)
        if trajectory_len == 0:
            return RewindResult(success=False, error="No trajectory history available")

        if steps <= 0:
            return RewindResult(success=False, error=f"Invalid steps: {steps}, must be positive")

        # Calculate waypoint indices
        start_idx = trajectory_len - 1
        end_idx = max(0, trajectory_len - 1 - steps)
        actual_steps = start_idx - end_idx

        if actual_steps == 0:
            return RewindResult(
                success=True,
                steps_rewound=0,
                start_waypoint_idx=start_idx,
                end_waypoint_idx=end_idx,
                error="Already at beginning of trajectory",
            )

        return await self._execute_rewind(start_idx, end_idx, dry_run, components)

    async def rewind_percentage(
        self,
        percentage: float,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Rewind the robot by a percentage of the trajectory.

        Args:
            percentage: Percentage of trajectory to rewind (0-100).
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        if percentage < 0 or percentage > 100:
            return RewindResult(
                success=False,
                error=f"Invalid percentage: {percentage}, must be 0-100",
            )

        trajectory_len = len(self._logger)
        if trajectory_len == 0:
            return RewindResult(success=False, error="No trajectory history available")

        steps = max(1, int(trajectory_len * percentage / 100))
        return await self.rewind_steps(steps, dry_run, components)

    async def rewind_to_safe(
        self,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Rewind to the last safe waypoint (inside workspace boundary).

        Args:
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        safe_idx = self.find_last_safe_waypoint()

        if safe_idx is None:
            return RewindResult(
                success=False,
                error="No safe waypoint found in trajectory history",
            )

        trajectory_len = len(self._logger)
        start_idx = trajectory_len - 1

        if safe_idx >= start_idx:
            return RewindResult(
                success=True,
                steps_rewound=0,
                start_waypoint_idx=start_idx,
                end_waypoint_idx=safe_idx,
                error="Already at or past the last safe waypoint",
            )

        return await self._execute_rewind(start_idx, safe_idx, dry_run, components)

    async def rewind_to_waypoint(
        self,
        waypoint_idx: int,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Rewind to a specific waypoint index.

        Args:
            waypoint_idx: Target waypoint index (0 = oldest).
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        trajectory_len = len(self._logger)

        if waypoint_idx < 0 or waypoint_idx >= trajectory_len:
            return RewindResult(
                success=False,
                error=f"Invalid waypoint index: {waypoint_idx}",
            )

        start_idx = trajectory_len - 1
        return await self._execute_rewind(start_idx, waypoint_idx, dry_run, components)

    async def reset_to_home(
        self,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Reset to home by rewinding 100% of the trajectory.

        Args:
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        logger.info("[RewindOrchestrator] Reset to home triggered")
        return await self.rewind_percentage(100.0, dry_run, components)

    # -------------------------------------------------------------------------
    # Rewind execution (chunked smooth version)
    # -------------------------------------------------------------------------

    def _interpolate_joints(
        self,
        q_start: List[float],
        q_end: List[float],
        t: float,
    ) -> List[float]:
        """Cubic interpolation between two joint configurations.

        Args:
            q_start: Starting joint positions.
            q_end: Ending joint positions.
            t: Interpolation parameter [0, 1].

        Returns:
            Interpolated joint positions.
        """
        # Cubic ease-in-out for smooth motion
        if t < 0.5:
            s = 4 * t * t * t
        else:
            s = 1 - (-2 * t + 2) ** 3 / 2

        return [qs + s * (qe - qs) for qs, qe in zip(q_start, q_end)]

    def _interpolate_waypoint_sequence(
        self,
        waypoints: List[UnifiedWaypoint],
        t: float,
    ) -> List[float]:
        """Interpolate through a sequence of waypoints.

        Args:
            waypoints: List of waypoints to interpolate through.
            t: Interpolation parameter [0, 1] for entire sequence.

        Returns:
            Interpolated joint positions.
        """
        if len(waypoints) < 2:
            return waypoints[0].arm_q if waypoints and waypoints[0].arm_q else []

        # Map t to segment index and local t
        n_segments = len(waypoints) - 1
        segment_t = t * n_segments
        segment_idx = min(int(segment_t), n_segments - 1)
        local_t = segment_t - segment_idx

        q_start = waypoints[segment_idx].arm_q
        q_end = waypoints[segment_idx + 1].arm_q

        if not q_start or not q_end:
            return q_start or q_end or []

        return self._interpolate_joints(q_start, q_end, local_t)

    async def _execute_rewind(
        self,
        start_idx: int,
        end_idx: int,
        dry_run: bool = False,
        components: Optional[List[str]] = None,
    ) -> RewindResult:
        """Execute the rewind operation from start_idx to end_idx.

        Uses chunked smooth interpolation for arm motion to reduce jitter.
        Waypoints are grouped into chunks, and the arm interpolates smoothly
        through each chunk while the base moves to the chunk endpoint.

        Args:
            start_idx: Starting waypoint index (more recent).
            end_idx: Ending waypoint index (older, target).
            dry_run: If True, only return what would happen.
            components: Components to rewind.

        Returns:
            RewindResult with operation details.
        """
        # Determine which components to rewind
        if components is None:
            components = []
            if self._config.rewind_base and self._base_backend:
                components.append("base")
            if self._config.rewind_arm and self._arm_backend:
                components.append("arm")
            if self._config.rewind_gripper and self._gripper_backend:
                components.append("gripper")

        # Build list of waypoints to traverse (in reverse order)
        waypoints_to_execute: List[UnifiedWaypoint] = []
        for i in range(start_idx, end_idx - 1, -1):
            if 0 <= i < len(self._logger):
                waypoints_to_execute.append(self._logger[i])

        if dry_run:
            return RewindResult(
                success=True,
                steps_rewound=len(waypoints_to_execute),
                start_waypoint_idx=start_idx,
                end_waypoint_idx=end_idx,
                waypoints_executed=[wp.to_dict() for wp in waypoints_to_execute],
                components_rewound=components,
            )

        self._is_rewinding = True
        executed_waypoints: List[Dict[str, Any]] = []

        try:
            chunk_size = self._config.chunk_size
            chunk_duration = self._config.chunk_duration
            n_waypoints = len(waypoints_to_execute)
            n_chunks = (n_waypoints + chunk_size - 1) // chunk_size  # Ceiling division

            logger.info(
                f"[RewindOrchestrator] Starting chunked rewind from waypoint {start_idx} to {end_idx} "
                f"({n_waypoints} waypoints in {n_chunks} chunks, components: {components})"
            )

            # Set arm to JOINT_POSITION mode before sending joint commands
            if "arm" in components and self._arm_backend:
                try:
                    self._arm_backend.set_control_mode(1)  # 1 = JOINT_POSITION
                    logger.info("[RewindOrchestrator] Set arm control mode to JOINT_POSITION")
                except Exception as e:
                    logger.warning(f"[RewindOrchestrator] Failed to set control mode: {e}")

            command_interval = 1.0 / self._config.command_rate

            # Process waypoints in chunks
            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, n_waypoints)
                chunk_waypoints = waypoints_to_execute[chunk_start:chunk_end]

                if not chunk_waypoints:
                    continue

                # Final waypoint of this chunk (target for base)
                final_wp = chunk_waypoints[-1]

                logger.info(
                    f"[RewindOrchestrator] Chunk {chunk_idx + 1}/{n_chunks} "
                    f"({len(chunk_waypoints)} waypoints)"
                )

                # Execute chunk with smooth arm interpolation
                await self._execute_chunk(
                    chunk_waypoints,
                    components,
                    chunk_duration,
                    command_interval,
                )

                # Record executed waypoints
                for wp in chunk_waypoints:
                    executed_waypoints.append(wp.to_dict())

                # Settle time between chunks (for base to catch up)
                if chunk_idx < n_chunks - 1 and self._config.settle_time > 0:
                    settle_end = asyncio.get_event_loop().time() + self._config.settle_time
                    while asyncio.get_event_loop().time() < settle_end:
                        # Keep sending final position to prevent arm timeout
                        if "arm" in components and self._arm_backend and final_wp.arm_q:
                            self._arm_backend.send_joint_position(final_wp.arm_q, blocking=False)
                        await asyncio.sleep(command_interval)

            logger.info(
                f"[RewindOrchestrator] Completed rewind: "
                f"{len(executed_waypoints)} waypoints executed in {n_chunks} chunks"
            )

            # Truncate trajectory to the target waypoint
            self._logger.truncate(end_idx + 1)

            return RewindResult(
                success=True,
                steps_rewound=len(executed_waypoints),
                start_waypoint_idx=start_idx,
                end_waypoint_idx=end_idx,
                waypoints_executed=executed_waypoints,
                components_rewound=components,
            )

        except Exception as e:
            logger.error(f"[RewindOrchestrator] Error during rewind: {e}")
            return RewindResult(
                success=False,
                steps_rewound=len(executed_waypoints),
                start_idx=start_idx,
                end_waypoint_idx=end_idx,
                error=str(e),
                waypoints_executed=executed_waypoints,
                components_rewound=components,
            )
        finally:
            self._is_rewinding = False

    def _interpolate_base_pose(
        self,
        waypoints: List[UnifiedWaypoint],
        t: float,
    ) -> tuple:
        """Interpolate base pose through a sequence of waypoints.

        Args:
            waypoints: List of waypoints to interpolate through.
            t: Interpolation parameter [0, 1] for entire sequence.

        Returns:
            Tuple of (x, y, theta).
        """
        if len(waypoints) < 2:
            wp = waypoints[0] if waypoints else None
            return (wp.x, wp.y, wp.theta) if wp else (0, 0, 0)

        # Map t to segment index and local t
        n_segments = len(waypoints) - 1
        segment_t = t * n_segments
        segment_idx = min(int(segment_t), n_segments - 1)
        local_t = segment_t - segment_idx

        wp_start = waypoints[segment_idx]
        wp_end = waypoints[segment_idx + 1]

        # Linear interpolation for base (Ruckig will smooth it)
        x = wp_start.x + local_t * (wp_end.x - wp_start.x)
        y = wp_start.y + local_t * (wp_end.y - wp_start.y)
        theta = wp_start.theta + local_t * (wp_end.theta - wp_start.theta)

        return (x, y, theta)

    async def _execute_chunk(
        self,
        chunk_waypoints: List[UnifiedWaypoint],
        components: List[str],
        duration: float,
        command_interval: float,
    ) -> None:
        """Execute a chunk of waypoints with smooth interpolation for both arm and base.

        Args:
            chunk_waypoints: Waypoints in this chunk.
            components: Components to command.
            duration: Total duration for this chunk (seconds).
            command_interval: Time between commands (seconds).
        """
        if not chunk_waypoints:
            return

        final_wp = chunk_waypoints[-1]

        # Send gripper to final position (no interpolation)
        if "gripper" in components and self._gripper_backend:
            self._gripper_backend.move(final_wp.gripper_position)

        # Interpolate both arm and base through all waypoints in chunk
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + duration

        while True:
            now = asyncio.get_event_loop().time()
            if now >= end_time:
                break

            # Calculate interpolation parameter [0, 1]
            t = (now - start_time) / duration
            t = min(max(t, 0.0), 1.0)

            # Interpolate and send base position
            if "base" in components and self._base_backend:
                x, y, theta = self._interpolate_base_pose(chunk_waypoints, t)
                self._base_backend.execute_action(x, y, theta)

            # Interpolate and send arm position
            if "arm" in components and self._arm_backend:
                q_interp = self._interpolate_waypoint_sequence(chunk_waypoints, t)
                if q_interp:
                    self._arm_backend.send_joint_position(q_interp, blocking=False)

            await asyncio.sleep(command_interval)

        # Ensure we end at the final position
        if "base" in components and self._base_backend:
            self._base_backend.execute_action(final_wp.x, final_wp.y, final_wp.theta)
        if "arm" in components and self._arm_backend and final_wp.arm_q:
            self._arm_backend.send_joint_position(final_wp.arm_q, blocking=False)

    async def _execute_waypoint(
        self,
        wp: UnifiedWaypoint,
        components: List[str],
    ) -> None:
        """Execute commands for a single waypoint.

        Args:
            wp: Waypoint to execute.
            components: Components to command.
        """
        # Base
        if "base" in components and self._base_backend:
            self._base_backend.execute_action(wp.x, wp.y, wp.theta)

        # Arm - use non-blocking (streaming) for faster command rate
        if "arm" in components and self._arm_backend and wp.arm_q:
            self._arm_backend.send_joint_position(wp.arm_q, blocking=False)

        # Gripper
        if "gripper" in components and self._gripper_backend:
            self._gripper_backend.move(wp.gripper_position)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current rewind status.

        Returns:
            Status dictionary.
        """
        return {
            "is_rewinding": self._is_rewinding,
            "trajectory_length": len(self._logger),
            "config": {
                "settle_time": self._config.settle_time,
                "command_rate": self._config.command_rate,
                "rewind_base": self._config.rewind_base,
                "rewind_arm": self._config.rewind_arm,
                "rewind_gripper": self._config.rewind_gripper,
                "safety_margin": self._config.safety_margin,
            },
            "backends": {
                "base": self._base_backend is not None,
                "arm": self._arm_backend is not None,
                "gripper": self._gripper_backend is not None,
            },
        }
