"""
Progress Tracker for thread-safe state management.
Maintains the current state of the pipeline for TUI rendering.
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class StepStatus(Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    ERROR = "error"
    SKIPPED = "skipped"


class WorkerState(Enum):
    """State of a worker thread."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class PipelineStep:
    """Represents a single step in the processing pipeline."""
    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    progress: float = 0.0  # 0-100
    current: int = 0
    total: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return None


@dataclass
class Worker:
    """Represents a worker thread."""
    id: int
    state: WorkerState = WorkerState.IDLE
    current_task: Optional[str] = None
    chunk_index: Optional[int] = None
    time_range: Optional[str] = None


@dataclass
class LogEntry:
    """Represents a log entry."""
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] [{self.level}] {self.message}"


class ProgressTracker:
    """Thread-safe progress tracker for the TUI."""

    def __init__(self):
        self._lock = Lock()

        # Pipeline state
        self.video_path: Optional[str] = None
        self.mode_info: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_step_index: int = 0
        self.steps: List[PipelineStep] = []

        # Worker state (for API mode)
        self.workers: List[Worker] = []

        # Queue state
        self.queue_completed: int = 0
        self.queue_in_progress: int = 0
        self.queue_pending: int = 0
        self.queue_failed: int = 0

        # Statistics
        self.total_segments: int = 0
        self.estimated_cost: float = 0.0
        self.processing_speed: Optional[float] = None  # e.g., 2.3x realtime

        # Logs
        self.logs: List[LogEntry] = []
        self.max_logs: int = 100

        # User actions
        self.paused: bool = False
        self.skip_correction: bool = False
        self.should_quit: bool = False

    def initialize_pipeline(self, video_path: str, mode_info: Dict[str, Any], steps: List[str]) -> None:
        """Initialize the pipeline with steps."""
        with self._lock:
            self.video_path = video_path
            self.mode_info = mode_info
            self.start_time = datetime.now()
            self.steps = [
                PipelineStep(name=f"step_{i}", description=step)
                for i, step in enumerate(steps, 1)
            ]

    def initialize_workers(self, count: int) -> None:
        """Initialize worker pool."""
        with self._lock:
            self.workers = [Worker(id=i) for i in range(count)]

    def start_step(self, step_index: int) -> None:
        """Mark a step as started."""
        with self._lock:
            if 0 <= step_index < len(self.steps):
                self.current_step_index = step_index
                step = self.steps[step_index]
                step.status = StepStatus.IN_PROGRESS
                step.start_time = datetime.now()

    def update_step_progress(
        self, step_index: int, current: int, total: int, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update step progress."""
        with self._lock:
            if 0 <= step_index < len(self.steps):
                step = self.steps[step_index]
                step.current = current
                step.total = total
                step.progress = (current / total * 100) if total > 0 else 0
                if metadata:
                    step.metadata.update(metadata)

    def complete_step(self, step_index: int) -> None:
        """Mark a step as complete."""
        with self._lock:
            if 0 <= step_index < len(self.steps):
                step = self.steps[step_index]
                step.status = StepStatus.COMPLETE
                step.end_time = datetime.now()
                step.progress = 100.0

    def error_step(self, step_index: int, error_message: str) -> None:
        """Mark a step as errored."""
        with self._lock:
            if 0 <= step_index < len(self.steps):
                step = self.steps[step_index]
                step.status = StepStatus.ERROR
                step.end_time = datetime.now()
                step.error_message = error_message

    def skip_step(self, step_index: int) -> None:
        """Mark a step as skipped."""
        with self._lock:
            if 0 <= step_index < len(self.steps):
                step = self.steps[step_index]
                step.status = StepStatus.SKIPPED
                step.end_time = datetime.now()

    def update_worker(self, worker_id: int, state: WorkerState, task_info: Optional[Dict[str, Any]] = None) -> None:
        """Update worker state."""
        with self._lock:
            if 0 <= worker_id < len(self.workers):
                worker = self.workers[worker_id]
                worker.state = state
                if task_info:
                    worker.current_task = task_info.get("task")
                    worker.chunk_index = task_info.get("chunk_index")
                    worker.time_range = task_info.get("time_range")
                elif state == WorkerState.IDLE:
                    worker.current_task = None
                    worker.chunk_index = None
                    worker.time_range = None

    def update_queue(self, completed: int = 0, in_progress: int = 0, pending: int = 0, failed: int = 0) -> None:
        """Update queue statistics."""
        with self._lock:
            if completed >= 0:
                self.queue_completed = completed
            if in_progress >= 0:
                self.queue_in_progress = in_progress
            if pending >= 0:
                self.queue_pending = pending
            if failed >= 0:
                self.queue_failed = failed

    def add_log(self, level: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a log entry."""
        with self._lock:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                message=message,
                metadata=metadata or {}
            )
            self.logs.append(entry)
            # Keep only last N logs
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]

    def update_stats(
        self,
        total_segments: Optional[int] = None,
        estimated_cost: Optional[float] = None,
        processing_speed: Optional[float] = None
    ) -> None:
        """Update statistics."""
        with self._lock:
            if total_segments is not None:
                self.total_segments = total_segments
            if estimated_cost is not None:
                self.estimated_cost = estimated_cost
            if processing_speed is not None:
                self.processing_speed = processing_speed

    def complete_pipeline(self) -> None:
        """Mark the entire pipeline as complete."""
        with self._lock:
            self.end_time = datetime.now()

    def get_state(self) -> Dict[str, Any]:
        """Get a snapshot of the current state (thread-safe)."""
        with self._lock:
            return {
                "video_path": self.video_path,
                "mode_info": self.mode_info.copy(),
                "start_time": self.start_time,
                "end_time": self.end_time,
                "current_step_index": self.current_step_index,
                "steps": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "status": s.status.value,
                        "progress": s.progress,
                        "current": s.current,
                        "total": s.total,
                        "duration": s.duration,
                        "error_message": s.error_message,
                        "metadata": s.metadata.copy()
                    }
                    for s in self.steps
                ],
                "workers": [
                    {
                        "id": w.id,
                        "state": w.state.value,
                        "current_task": w.current_task,
                        "chunk_index": w.chunk_index,
                        "time_range": w.time_range
                    }
                    for w in self.workers
                ],
                "queue": {
                    "completed": self.queue_completed,
                    "in_progress": self.queue_in_progress,
                    "pending": self.queue_pending,
                    "failed": self.queue_failed
                },
                "stats": {
                    "total_segments": self.total_segments,
                    "estimated_cost": self.estimated_cost,
                    "processing_speed": self.processing_speed
                },
                "logs": [str(log) for log in self.logs[-20:]],  # Last 20 logs
                "user_actions": {
                    "paused": self.paused,
                    "skip_correction": self.skip_correction,
                    "should_quit": self.should_quit
                }
            }

    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if self.start_time:
            end = self.end_time or datetime.now()
            return (end - self.start_time).total_seconds()
        return None

    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimate time remaining based on current progress."""
        if not self.start_time or self.current_step_index >= len(self.steps):
            return None

        completed_steps = sum(1 for s in self.steps if s.status == StepStatus.COMPLETE)
        if completed_steps == 0:
            return None

        elapsed = self.elapsed_time
        if elapsed is None:
            return None

        total_steps = len(self.steps)
        avg_time_per_step = elapsed / completed_steps
        remaining_steps = total_steps - completed_steps

        # Factor in current step progress
        current_step = self.steps[self.current_step_index]
        if current_step.progress > 0:
            current_step_remaining = (100 - current_step.progress) / 100
            return (remaining_steps - 1 + current_step_remaining) * avg_time_per_step

        return remaining_steps * avg_time_per_step
