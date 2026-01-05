"""
Custom Textual widgets for the Subscene Generator TUI.
"""

from textual.widgets import Static, ProgressBar, Label
from textual.containers import Container, Vertical, Horizontal
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from typing import List, Dict, Any, Optional
from datetime import datetime


class HeaderWidget(Static):
    """Header widget showing video info and overall progress."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._video_path = ""
        self._mode_info = {}
        self._start_time = None

    def update_info(self, video_path: str, mode_info: Dict[str, Any], start_time: datetime):
        """Update header information."""
        self._video_path = video_path
        self._mode_info = mode_info
        self._start_time = start_time
        self.refresh()

    def render(self) -> Panel:
        """Render the header."""
        # Format video name (just filename)
        video_name = self._video_path.split("/")[-1] if self._video_path else "N/A"

        # Format mode info
        mode_parts = []
        if self._mode_info.get("local_whisper"):
            model = self._mode_info.get("whisper_model", "unknown")
            device = self._mode_info.get("device", "unknown")
            mode_parts.append(f"Local Whisper ({model}, {device})")
        else:
            mode_parts.append("API Whisper")

        translator = self._mode_info.get("translator", "openai")
        mode_parts.append(translator.upper())

        if self._mode_info.get("correction_enabled"):
            mode_parts.append("with Correction")

        mode_str = " + ".join(mode_parts)

        # Format elapsed time
        elapsed_str = "00:00:00"
        eta_str = "calculating..."
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        title = Text("Subscene Generator - Whisper Subtitle Pipeline", style="bold cyan")
        content = Text()
        content.append(f"Video: {video_name}\n", style="yellow")
        content.append(f"Mode: {mode_str}\n", style="green")
        content.append(f"Elapsed: {elapsed_str} | Estimated remaining: {eta_str}", style="magenta")

        return Panel(content, title=title, border_style="cyan")


class PipelineProgressWidget(Static):
    """Widget showing the pipeline steps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._steps_data = []

    def update_steps(self, steps: List[Dict[str, Any]]):
        """Update pipeline steps."""
        self._steps_data = steps
        self.refresh()

    def render(self) -> Panel:
        """Render the pipeline progress."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Status", width=3)
        table.add_column("Step", ratio=1)
        table.add_column("Duration", width=10, justify="right")

        for i, step in enumerate(self._steps_data, 1):
            status = step["status"]
            description = step["description"]
            duration = step.get("duration")

            # Status icon
            if status == "complete":
                icon = "[green]✓[/green]"
            elif status == "in_progress":
                icon = "[yellow]●[/yellow]"
            elif status == "error":
                icon = "[red]✗[/red]"
            elif status == "skipped":
                icon = "[dim]○[/dim]"
            else:  # pending
                icon = "[dim]○[/dim]"

            # Format step text
            if status == "in_progress":
                progress = step.get("progress", 0)
                current = step.get("current", 0)
                total = step.get("total", 0)
                if total > 0:
                    step_text = f"[bold]{i}. {description}[/bold] ({current}/{total}, {progress:.0f}%)"
                else:
                    step_text = f"[bold]{i}. {description}[/bold]"
            elif status == "error":
                step_text = f"[red]{i}. {description} - {step.get('error_message', 'Failed')}[/red]"
            else:
                step_text = f"{i}. {description}"

            # Format duration
            if duration:
                dur_str = f"({duration:.1f}s)" if duration < 60 else f"({duration/60:.1f}m)"
            else:
                dur_str = ""

            table.add_row(icon, step_text, dur_str)

        return Panel(table, title="[bold]Pipeline Progress[/bold]", border_style="blue")


class CurrentStepWidget(Static):
    """Widget showing detailed progress for the current step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_step_data = None

    def update_current_step(self, step: Optional[Dict[str, Any]]):
        """Update current step information."""
        self._current_step_data = step
        self.refresh()

    def render(self) -> Panel:
        """Render the current step details."""
        if not self._current_step_data or self._current_step_data["status"] not in ["in_progress"]:
            return Panel("[dim]Waiting for next step...[/dim]", title="[bold]Current Step[/bold]", border_style="yellow")

        description = self._current_step_data["description"]
        progress = self._current_step_data.get("progress", 0)
        current = self._current_step_data.get("current", 0)
        total = self._current_step_data.get("total", 0)
        metadata = self._current_step_data.get("metadata", {})

        # Progress bar
        bar_width = 50
        filled = int(bar_width * progress / 100)
        empty = bar_width - filled
        bar = "▰" * filled + "▱" * empty

        content = Text()
        content.append(f"{description}\n", style="bold yellow")

        if total > 0:
            content.append(f"{bar} {progress:.0f}% ({current}/{total})\n", style="cyan")
        else:
            content.append(f"{bar} {progress:.0f}%\n", style="cyan")

        # Add metadata
        if "speed" in metadata:
            content.append(f"Processing speed: {metadata['speed']:.1f}x realtime\n", style="green")
        if "segments" in metadata:
            content.append(f"Segments generated: {metadata['segments']}\n", style="green")

        return Panel(content, title=f"[bold]Current Step: {description}[/bold]", border_style="yellow")


class WorkerStatusWidget(Static):
    """Widget showing worker status (for API mode)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._workers_data = []
        self._queue_data = {}

    def update_workers(self, workers: List[Dict[str, Any]], queue: Dict[str, int]):
        """Update worker status."""
        self._workers_data = workers
        self._queue_data = queue
        self.refresh()

    def render(self) -> Panel:
        """Render worker status."""
        if not self._workers_data:
            return Panel("[dim]N/A (Local Whisper - single threaded)[/dim]", title="[bold]Workers[/bold]", border_style="green")

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Worker", width=8)
        table.add_column("Status", width=8)
        table.add_column("Task", ratio=1)

        for worker in self._workers_data:
            worker_id = f"Worker {worker['id']}"
            state = worker["state"]

            if state == "busy":
                status = "[yellow]BUSY[/yellow]"
                task_info = worker.get("current_task", "Unknown task")
                if worker.get("time_range"):
                    task_info += f" ({worker['time_range']})"
            elif state == "error":
                status = "[red]ERROR[/red]"
                task_info = "Failed"
            else:  # idle
                status = "[dim]IDLE[/dim]"
                task_info = "Waiting for chunks..."

            table.add_row(worker_id, status, task_info)

        # Add queue info
        content = Text()
        content.append(table)
        content.append("\n")
        completed = self._queue_data.get("completed", 0)
        in_progress = self._queue_data.get("in_progress", 0)
        pending = self._queue_data.get("pending", 0)
        failed = self._queue_data.get("failed", 0)
        content.append(
            f"Queue: Completed: {completed} | In Progress: {in_progress} | Pending: {pending} | Failed: {failed}",
            style="cyan"
        )

        return Panel(content, title=f"[bold]Workers ({len(self._workers_data)} active)[/bold]", border_style="green")


class LogWidget(Static):
    """Widget showing recent activity logs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logs_data = []

    def update_logs(self, logs: List[str]):
        """Update logs."""
        self._logs_data = logs
        self.refresh()

    def render(self) -> Panel:
        """Render logs."""
        if not self._logs_data:
            return Panel("[dim]No activity yet...[/dim]", title="[bold]Recent Activity[/bold]", border_style="magenta")

        content = Text()
        for log in self._logs_data[-10:]:  # Show last 10 logs
            # Parse log level for styling
            if "[ERROR]" in log:
                content.append(log + "\n", style="red")
            elif "[WARNING]" in log:
                content.append(log + "\n", style="yellow")
            elif "[INFO]" in log:
                content.append(log + "\n", style="blue")
            else:
                content.append(log + "\n")

        return Panel(content, title="[bold]Recent Activity[/bold]", border_style="magenta")


class StatsWidget(Static):
    """Widget showing statistics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stats_data = {}

    def update_stats(self, stats: Dict[str, Any]):
        """Update statistics."""
        self._stats_data = stats
        self.refresh()

    def render(self) -> Panel:
        """Render statistics."""
        segments = self._stats_data.get("total_segments", 0)
        cost = self._stats_data.get("estimated_cost", 0.0)
        speed = self._stats_data.get("processing_speed")

        content = Text()
        content.append(f"Segments generated: {segments}", style="cyan")

        if cost > 0:
            content.append(f" | Estimated cost: ${cost:.2f}", style="green")
        else:
            content.append(f" | Estimated cost: $0.00 (local)", style="green")

        if speed:
            content.append(f" | Processing speed: {speed:.1f}x realtime", style="yellow")

        return Panel(content, title="[bold]Stats[/bold]", border_style="white")


class HelpWidget(Static):
    """Widget showing keyboard shortcuts."""

    def render(self) -> Text:
        """Render help text."""
        content = Text()
        content.append("[dim]Keyboard: [/dim]")
        content.append("[cyan]q[/cyan] Quit ", style="dim")
        content.append("[cyan]p[/cyan] Pause/Resume ", style="dim")
        content.append("[cyan]s[/cyan] Skip Correction ", style="dim")
        content.append("[cyan]l[/cyan] Toggle Log Level ", style="dim")
        content.append("[cyan]/[/cyan] Search", style="dim")
        return content
