"""Health monitoring for trading engine.

Tracks system health metrics including:
- Uptime and runtime statistics
- Event processing rates
- Error tracking
- Performance metrics
- Heartbeat generation
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque


@dataclass
class HealthMetrics:
    """Container for health monitoring metrics."""

    start_time: float
    last_heartbeat: float
    total_events: int = 0
    total_fills: int = 0
    error_count: int = 0
    warning_count: int = 0

    # Performance metrics
    event_processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    events_per_second: float = 0.0
    avg_event_time_ms: float = 0.0

    # Resource metrics
    peak_memory_mb: float = 0.0

    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        return time.time() - self.start_time

    def uptime_formatted(self) -> str:
        """Format uptime as human-readable string."""
        total_seconds = int(self.uptime_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class HealthMonitor:
    """Monitors trading engine health and performance.

    Provides real-time health status, performance metrics, and
    heartbeat generation for monitoring systems.

    Parameters:
    ----------
    heartbeat_interval : float
        Seconds between heartbeat emissions (default: 60.0)
    """

    def __init__(self, heartbeat_interval: float = 60.0) -> None:
        self.heartbeat_interval = heartbeat_interval
        self.metrics = HealthMetrics(
            start_time=time.time(),
            last_heartbeat=time.time()
        )

        # Recent errors for debugging
        self.recent_errors: deque = deque(maxlen=100)
        self.recent_warnings: deque = deque(maxlen=100)

        # Custom metric tracking
        self.custom_metrics: Dict[str, float] = {}

    def record_event(self, processing_time_ms: Optional[float] = None) -> None:
        """Record an event processing completion.

        Parameters:
        ----------
        processing_time_ms : Optional[float]
            Processing time in milliseconds
        """
        self.metrics.total_events += 1

        if processing_time_ms is not None:
            self.metrics.event_processing_times.append(processing_time_ms)

            # Update average
            if len(self.metrics.event_processing_times) > 0:
                self.metrics.avg_event_time_ms = (
                    sum(self.metrics.event_processing_times) /
                    len(self.metrics.event_processing_times)
                )

    def record_fill(self) -> None:
        """Record a trade fill."""
        self.metrics.total_fills += 1

    def record_error(self, error_msg: str, error_type: Optional[str] = None) -> None:
        """Record an error.

        Parameters:
        ----------
        error_msg : str
            Error message
        error_type : Optional[str]
            Error type/category
        """
        self.metrics.error_count += 1
        self.recent_errors.append({
            "timestamp": time.time(),
            "message": error_msg,
            "type": error_type or "unknown"
        })

    def record_warning(self, warning_msg: str) -> None:
        """Record a warning.

        Parameters:
        ----------
        warning_msg : str
            Warning message
        """
        self.metrics.warning_count += 1
        self.recent_warnings.append({
            "timestamp": time.time(),
            "message": warning_msg
        })

    def set_custom_metric(self, name: str, value: float) -> None:
        """Set a custom metric value.

        Parameters:
        ----------
        name : str
            Metric name
        value : float
            Metric value
        """
        self.custom_metrics[name] = value

    def update_events_per_second(self, window_seconds: float = 60.0) -> None:
        """Update events per second calculation.

        Parameters:
        ----------
        window_seconds : float
            Time window for rate calculation
        """
        uptime = self.metrics.uptime_seconds()
        if uptime > 0:
            # Use shorter of uptime or window
            actual_window = min(uptime, window_seconds)
            self.metrics.events_per_second = self.metrics.total_events / actual_window

    def should_emit_heartbeat(self) -> bool:
        """Check if it's time to emit a heartbeat.

        Returns:
        -------
        bool
            True if heartbeat should be emitted
        """
        elapsed = time.time() - self.metrics.last_heartbeat
        return elapsed >= self.heartbeat_interval

    def emit_heartbeat(self) -> Dict:
        """Emit heartbeat with current metrics.

        Returns:
        -------
        Dict
            Heartbeat data
        """
        self.metrics.last_heartbeat = time.time()
        self.update_events_per_second()

        return self.get_status()

    def get_status(self) -> Dict:
        """Get current health status.

        Returns:
        -------
        Dict
            Health status dictionary
        """
        status = {
            "timestamp": time.time(),
            "uptime_seconds": self.metrics.uptime_seconds(),
            "uptime_formatted": self.metrics.uptime_formatted(),
            "total_events": self.metrics.total_events,
            "total_fills": self.metrics.total_fills,
            "error_count": self.metrics.error_count,
            "warning_count": self.metrics.warning_count,
            "events_per_second": self.metrics.events_per_second,
            "avg_event_time_ms": self.metrics.avg_event_time_ms,
            "healthy": self._is_healthy(),
            "custom_metrics": self.custom_metrics.copy(),
        }

        return status

    def get_recent_errors(self, count: int = 10) -> List[Dict]:
        """Get recent errors.

        Parameters:
        ----------
        count : int
            Number of recent errors to return

        Returns:
        -------
        List[Dict]
            Recent error records
        """
        return list(self.recent_errors)[-count:]

    def get_recent_warnings(self, count: int = 10) -> List[Dict]:
        """Get recent warnings.

        Parameters:
        ----------
        count : int
            Number of recent warnings to return

        Returns:
        -------
        List[Dict]
            Recent warning records
        """
        return list(self.recent_warnings)[-count:]

    def _is_healthy(self) -> bool:
        """Determine if system is healthy.

        Returns:
        -------
        bool
            True if system is healthy
        """
        # Simple health check: low error rate
        uptime = self.metrics.uptime_seconds()

        # Must be running for at least 10 seconds
        if uptime < 10.0:
            return True

        # Check error rate (errors per minute)
        error_rate = (self.metrics.error_count / uptime) * 60.0

        # Healthy if less than 5 errors per minute
        return error_rate < 5.0

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = HealthMetrics(
            start_time=time.time(),
            last_heartbeat=time.time()
        )
        self.recent_errors.clear()
        self.recent_warnings.clear()
        self.custom_metrics.clear()

    def __str__(self) -> str:
        """String representation."""
        return (
            f"HealthMonitor(uptime={self.metrics.uptime_formatted()}, "
            f"events={self.metrics.total_events}, "
            f"fills={self.metrics.total_fills}, "
            f"errors={self.metrics.error_count}, "
            f"healthy={self._is_healthy()})"
        )
