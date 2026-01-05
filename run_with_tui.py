#!/usr/bin/env python3
"""
Wrapper to run main.py processing in a background thread while TUI runs in main thread.
This is a temporary solution to test the TUI - the proper fix would be to refactor main.py.
"""

import sys
import threading
from pathlib import Path

# Import the TUI
from tui.app import TUIManager

def run_processing(args):
    """Run the main processing in a background thread."""
    # Import main after we're in the thread to avoid issues
    import main as main_module

    # Temporarily replace sys.argv with our args
    old_argv = sys.argv
    sys.argv = ['main.py'] + args

    try:
        main_module.main()
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_with_tui.py <video_file> [options]")
        print("Example: python run_with_tui.py test.mp4 --local-whisper --model tiny")
        sys.exit(1)

    # Check if TUI should be disabled
    if "--no-tui" in sys.argv:
        # Just run main.py directly
        import main
        main.main()
    else:
        # Start processing in background thread
        args = sys.argv[1:]
        processing_thread = threading.Thread(
            target=run_processing,
            args=(args,),
            daemon=True
        )

        # Create and run TUI in main thread
        print("Starting TUI...")
        import time
        time.sleep(0.5)  # Give imports time

        processing_thread.start()

        # Run TUI (this will block until quit)
        tui_manager = TUIManager()
        tui_manager.run()

        # Wait for processing to finish
        processing_thread.join(timeout=5)
