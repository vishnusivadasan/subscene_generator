"""
tqdm wrapper that can be disabled when TUI is active
"""

from tqdm import tqdm as original_tqdm
import os

# Check if TUI mode is active via environment variable
TUI_ACTIVE = os.getenv('SUBSCENE_TUI_MODE') == '1'

class tqdm_or_dummy:
    """Wrapper that returns dummy tqdm when TUI is active"""

    def __init__(self, *args, **kwargs):
        if TUI_ACTIVE:
            # Return dummy that does nothing
            self._pbar = None
            self._iterable = args[0] if args else None
        else:
            self._pbar = original_tqdm(*args, **kwargs)
            self._iterable = None

    def __enter__(self):
        if self._pbar:
            return self._pbar.__enter__()
        return self

    def __exit__(self, *args):
        if self._pbar:
            return self._pbar.__exit__(*args)

    def __iter__(self):
        if self._pbar:
            return iter(self._pbar)
        elif self._iterable:
            return iter(self._iterable)
        return iter([])

    def update(self, n=1):
        if self._pbar:
            self._pbar.update(n)

    def close(self):
        if self._pbar:
            self._pbar.close()

    def set_description(self, desc):
        if self._pbar:
            self._pbar.set_description(desc)

    def refresh(self):
        if self._pbar:
            self._pbar.refresh()

    @property
    def total(self):
        if self._pbar:
            return self._pbar.total
        return 0

    @total.setter
    def total(self, value):
        if self._pbar:
            self._pbar.total = value


# Export as tqdm
tqdm = tqdm_or_dummy
