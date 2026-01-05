"""
Main Textual application for Subscene Generator TUI.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Header, Footer
from textual import work
from typing import Optional
import asyncio

from .widgets import (
    HeaderWidget,
    PipelineProgressWidget,
    CurrentStepWidget,
    WorkerStatusWidget,
    LogWidget,
    StatsWidget,
    HelpWidget
)
from .progress_tracker import ProgressTracker, StepStatus
from .event_bus import event_bus, Events


class SubsceneApp(App):
    """Textual app for Subscene Generator."""

    CSS_PATH = "styles.css"
    BINDINGS = [
        ("q", "quit_app", "Quit"),
        ("p", "toggle_pause", "Pause/Resume"),
        ("s", "skip_correction", "Skip Correction"),
        ("l", "toggle_log_level", "Toggle Log Level"),
    ]

    def __init__(self, progress_tracker: ProgressTracker, **kwargs):
        super().__init__(**kwargs)
        self.tracker = progress_tracker
        self.update_interval = 0.5  # Update every 500ms

        # Widgets (will be populated in compose)
        self.header_widget = None
        self.pipeline_widget = None
        self.current_step_widget = None
        self.worker_widget = None
        self.log_widget = None
        self.stats_widget = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        self.header_widget = HeaderWidget(id="header")
        self.pipeline_widget = PipelineProgressWidget(id="pipeline")
        self.current_step_widget = CurrentStepWidget(id="current_step")
        self.worker_widget = WorkerStatusWidget(id="workers")
        self.log_widget = LogWidget(id="logs")
        self.stats_widget = StatsWidget(id="stats")
        help_widget = HelpWidget(id="help")

        yield Header(show_clock=True)
        yield Container(
            self.header_widget,
            self.pipeline_widget,
            self.current_step_widget,
            self.worker_widget,
            ScrollableContainer(self.log_widget, id="log_container"),
            self.stats_widget,
            help_widget,
            id="main_container"
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.run_worker(self.update_ui_loop(), exclusive=True)

    @work(exclusive=True)
    async def update_ui_loop(self) -> None:
        """Continuously update the UI from tracker state."""
        while True:
            try:
                state = self.tracker.get_state()
                self.update_widgets(state)
                await asyncio.sleep(self.update_interval)

                # Check if pipeline is complete
                if state.get("end_time"):
                    # Give user time to see final state before exit
                    await asyncio.sleep(2)
                    # Don't auto-exit, let user press 'q'

                # Check if user requested quit
                if self.tracker.should_quit:
                    self.exit()

            except Exception as e:
                self.tracker.add_log("ERROR", f"UI update error: {e}")

    def update_widgets(self, state: dict) -> None:
        """Update all widgets with current state."""
        try:
            # Update header
            if self.header_widget and state.get("video_path"):
                self.header_widget.update_info(
                    state["video_path"],
                    state["mode_info"],
                    state["start_time"]
                )

            # Update pipeline
            if self.pipeline_widget:
                self.pipeline_widget.update_steps(state["steps"])

            # Update current step
            if self.current_step_widget:
                current_idx = state.get("current_step_index", 0)
                if 0 <= current_idx < len(state["steps"]):
                    current_step = state["steps"][current_idx]
                    self.current_step_widget.update_current_step(current_step)
                else:
                    self.current_step_widget.update_current_step(None)

            # Update workers
            if self.worker_widget:
                self.worker_widget.update_workers(
                    state.get("workers", []),
                    state.get("queue", {})
                )

            # Update logs
            if self.log_widget:
                self.log_widget.update_logs(state.get("logs", []))

            # Update stats
            if self.stats_widget:
                self.stats_widget.update_stats(state.get("stats", {}))

        except Exception as e:
            # Don't crash the app on widget update errors
            self.tracker.add_log("ERROR", f"Widget update error: {e}")

    def action_quit_app(self) -> None:
        """Handle quit action."""
        # Check if pipeline is still running
        state = self.tracker.get_state()
        if not state.get("end_time"):
            # Confirm quit
            self.tracker.should_quit = True
            self.tracker.add_log("WARNING", "Quit requested - stopping after current operation...")
        else:
            self.exit()

    def action_toggle_pause(self) -> None:
        """Handle pause/resume action."""
        self.tracker.paused = not self.tracker.paused
        status = "paused" if self.tracker.paused else "resumed"
        self.tracker.add_log("INFO", f"Pipeline {status}")
        event_bus.emit(Events.USER_PAUSE if self.tracker.paused else Events.USER_RESUME)

    def action_skip_correction(self) -> None:
        """Handle skip correction action."""
        if not self.tracker.skip_correction:
            self.tracker.skip_correction = True
            self.tracker.add_log("INFO", "Correction step will be skipped")
            event_bus.emit(Events.USER_SKIP_CORRECTION)

    def action_toggle_log_level(self) -> None:
        """Handle toggle log level action."""
        # This could cycle through different log verbosity levels
        self.tracker.add_log("INFO", "Log level toggle not yet implemented")


class TUIManager:
    """Manager for running the TUI app in a separate thread."""

    def __init__(self):
        self.tracker = ProgressTracker()
        self.app: Optional[SubsceneApp] = None
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup event handlers to update tracker from events."""
        # Pipeline events
        event_bus.subscribe(Events.PIPELINE_START, self._handle_pipeline_start)
        event_bus.subscribe(Events.PIPELINE_COMPLETE, self._handle_pipeline_complete)
        event_bus.subscribe(Events.STEP_START, self._handle_step_start)
        event_bus.subscribe(Events.STEP_COMPLETE, self._handle_step_complete)
        event_bus.subscribe(Events.STEP_PROGRESS, self._handle_step_progress)

        # Transcription events
        event_bus.subscribe(Events.TRANSCRIBE_START, self._handle_transcribe_start)
        event_bus.subscribe(Events.TRANSCRIBE_PROGRESS, self._handle_transcribe_progress)
        event_bus.subscribe(Events.WORKER_STATUS, self._handle_worker_status)

        # Log events
        event_bus.subscribe(Events.LOG_INFO, self._handle_log)
        event_bus.subscribe(Events.LOG_WARNING, self._handle_log)
        event_bus.subscribe(Events.LOG_ERROR, self._handle_log)
        event_bus.subscribe(Events.LOG_DEBUG, self._handle_log)

    def _handle_pipeline_start(self, event_type: str, data: dict):
        """Handle pipeline start event."""
        self.tracker.initialize_pipeline(
            data["video_path"],
            data["mode_info"],
            data["steps"]
        )
        if data.get("workers", 0) > 0:
            self.tracker.initialize_workers(data["workers"])

    def _handle_pipeline_complete(self, event_type: str, data: dict):
        """Handle pipeline complete event."""
        self.tracker.complete_pipeline()

    def _handle_step_start(self, event_type: str, data: dict):
        """Handle step start event."""
        self.tracker.start_step(data["step_index"])

    def _handle_step_complete(self, event_type: str, data: dict):
        """Handle step complete event."""
        self.tracker.complete_step(data["step_index"])

    def _handle_step_progress(self, event_type: str, data: dict):
        """Handle step progress event."""
        self.tracker.update_step_progress(
            data["step_index"],
            data["current"],
            data["total"],
            data.get("metadata")
        )

    def _handle_transcribe_start(self, event_type: str, data: dict):
        """Handle transcribe start event."""
        if data.get("workers"):
            self.tracker.initialize_workers(data["workers"])

    def _handle_transcribe_progress(self, event_type: str, data: dict):
        """Handle transcribe progress event."""
        self.tracker.update_queue(
            completed=data.get("completed", 0),
            in_progress=data.get("in_progress", 0),
            pending=data.get("pending", 0),
            failed=data.get("failed", 0)
        )

    def _handle_worker_status(self, event_type: str, data: dict):
        """Handle worker status event."""
        from .progress_tracker import WorkerState
        state_map = {
            "idle": WorkerState.IDLE,
            "busy": WorkerState.BUSY,
            "error": WorkerState.ERROR
        }
        self.tracker.update_worker(
            data["worker_id"],
            state_map.get(data["state"], WorkerState.IDLE),
            data.get("task_info")
        )

    def _handle_log(self, event_type: str, data: dict):
        """Handle log event."""
        level_map = {
            Events.LOG_INFO: "INFO",
            Events.LOG_WARNING: "WARNING",
            Events.LOG_ERROR: "ERROR",
            Events.LOG_DEBUG: "DEBUG"
        }
        level = level_map.get(event_type, "INFO")
        self.tracker.add_log(level, data["message"], data.get("metadata"))

    def run(self):
        """Run the TUI app."""
        self.app = SubsceneApp(self.tracker)
        self.app.run()

    def is_paused(self) -> bool:
        """Check if user requested pause."""
        return self.tracker.paused

    def should_skip_correction(self) -> bool:
        """Check if user requested to skip correction."""
        return self.tracker.skip_correction

    def should_quit(self) -> bool:
        """Check if user requested quit."""
        return self.tracker.should_quit
