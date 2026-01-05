#!/usr/bin/env python3
"""
TUI-enabled wrapper for main.py
Runs processing in background thread while TUI runs in main thread.
"""

import sys
import threading
import time
import os

# Set environment variable to disable tqdm
os.environ['SUBSCENE_TUI_MODE'] = '1'

# Import main processing
import main

# Import TUI
from tui.app import TUIManager
from tui.event_bus import Events, event_bus

def run_main_in_thread(argv):
    """Run main.py in a background thread."""
    # Override sys.argv for main.py
    old_argv = sys.argv
    sys.argv = argv

    try:
        # Give TUI time to start
        time.sleep(1)
        main.main()
    except KeyboardInterrupt:
        event_bus.emit(Events.PIPELINE_ERROR, {"error": "Interrupted"})
    except Exception as e:
        event_bus.emit(Events.PIPELINE_ERROR, {"error": str(e)})
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = old_argv
        # Signal TUI that processing is done
        time.sleep(2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_tui.py <video_file> [options]")
        print("Example: python main_tui.py test.mp4 --local-whisper --model tiny")
        print("\nNote: Adds --no-tui is not supported in this mode.")
        sys.exit(1)

    # Check TTY
    if not sys.stdout.isatty():
        print("Error: TUI requires an interactive terminal")
        print("Run 'python main.py' instead for non-TTY mode")
        sys.exit(1)

    # Prepare argv for main.py (add --no-tui to disable conflicting tqdm)
    main_argv = ["main.py"] + sys.argv[1:]

    # Start processing thread
    print("Starting subtitle generator with TUI...")
    processing_thread = threading.Thread(
        target=run_main_in_thread,
        args=(main_argv,),
        daemon=True
    )
    processing_thread.start()

    # Run TUI in main thread
    try:
        tui_manager = TUIManager()
        tui_manager.run()
    except KeyboardInterrupt:
        print("\nTUI interrupted")
    finally:
        # Wait a bit for processing thread
        processing_thread.join(timeout=3)
        print("TUI closed")
