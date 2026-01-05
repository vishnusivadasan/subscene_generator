"""
Event Bus for communication between processing functions and TUI.
Thread-safe pub/sub system for real-time progress updates.
"""

from threading import Lock
from typing import Callable, Dict, List, Any
from collections import defaultdict


class EventBus:
    """Thread-safe event bus for publishing and subscribing to events."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = Lock()
        self._initialized = True

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        with self._lock:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].remove(callback)

    def emit(self, event_type: str, data: Any = None) -> None:
        """Emit an event to all subscribers."""
        with self._lock:
            callbacks = self._subscribers.get(event_type, []).copy()

        for callback in callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                # Don't let subscriber errors crash the event bus
                print(f"Error in event callback for {event_type}: {e}")

    def clear(self) -> None:
        """Clear all subscribers (useful for testing)."""
        with self._lock:
            self._subscribers.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None


# Singleton instance
event_bus = EventBus()


# Event type constants
class Events:
    """Event type constants for type safety and documentation."""

    # Pipeline events
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR = "pipeline_error"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    STEP_PROGRESS = "step_progress"

    # Audio extraction events
    AUDIO_EXTRACT_START = "audio_extract_start"
    AUDIO_EXTRACT_COMPLETE = "audio_extract_complete"
    AUDIO_EXTRACT_ERROR = "audio_extract_error"

    # Chunking events
    CHUNK_INFO = "chunk_info"
    CHUNK_CREATED = "chunk_created"
    CHUNK_COMPLETE = "chunk_complete"

    # Transcription events (API)
    TRANSCRIBE_START = "transcribe_start"
    TRANSCRIBE_PROGRESS = "transcribe_progress"
    TRANSCRIBE_COMPLETE = "transcribe_complete"
    TRANSCRIBE_ERROR = "transcribe_error"
    TRANSCRIBE_RETRY = "transcribe_retry"
    WORKER_STATUS = "worker_status"

    # Transcription events (Local)
    LOCAL_MODEL_LOADING = "local_model_loading"
    LOCAL_MODEL_LOADED = "local_model_loaded"
    LOCAL_LANGUAGE_DETECT = "local_language_detect"
    LOCAL_TRANSCRIBE_PROGRESS = "local_transcribe_progress"

    # Translation events
    TRANSLATE_START = "translate_start"
    TRANSLATE_PROGRESS = "translate_progress"
    TRANSLATE_COMPLETE = "translate_complete"
    TRANSLATE_BATCH_START = "translate_batch_start"
    TRANSLATE_BATCH_COMPLETE = "translate_batch_complete"
    TRANSLATE_FALLBACK = "translate_fallback"
    TRANSLATE_ERROR = "translate_error"

    # Correction events
    CORRECT_START = "correct_start"
    CORRECT_PROGRESS = "correct_progress"
    CORRECT_COMPLETE = "correct_complete"
    CORRECT_ERROR = "correct_error"

    # SRT generation events
    SRT_START = "srt_start"
    SRT_COMPLETE = "srt_complete"

    # Log events
    LOG_INFO = "log_info"
    LOG_WARNING = "log_warning"
    LOG_ERROR = "log_error"
    LOG_DEBUG = "log_debug"

    # User action events
    USER_PAUSE = "user_pause"
    USER_RESUME = "user_resume"
    USER_SKIP_CORRECTION = "user_skip_correction"
    USER_QUIT = "user_quit"
