"""Reliable Ctrl-C handling.

A normal Ctrl-C raises KeyboardInterrupt in the main thread, which is what we want. But if
the main thread is stuck (a hung network call, a handler that swallows the interrupt) it can
look like Ctrl-C does nothing. So a watcher thread is also notified, at the C level, the
instant a signal arrives (``signal.set_wakeup_fd``), whatever the main thread is doing:

  1st Ctrl-C   main thread gets KeyboardInterrupt as usual; a grace timer starts
  2nd Ctrl-C   exit immediately
  grace timer  exit anyway if cleanup has not finished within ``grace_s`` seconds

Both forced paths restore the terminal first (leave the full-screen dashboard, show the cursor).
"""

from __future__ import annotations

import os
import signal
import threading

_TERMINAL_RESET = b"\x1b[?1049l\x1b[?25h\x1b[0m"      # leave alt screen, show cursor, reset colours


def hard_exit(code: int = 130) -> None:
    """Restore the terminal and exit without waiting for anything."""
    try:
        if os.isatty(1):
            os.write(1, _TERMINAL_RESET)
        os.write(2, b"\nForced exit.\n")
    finally:
        os._exit(code)


def _raise_keyboard_interrupt(signum, frame):          # SIGTERM behaves like Ctrl-C
    raise KeyboardInterrupt


def install(grace_s: float = 6.0) -> None:
    """Call from the main thread."""
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    def watch() -> None:
        presses = 0
        while True:
            for signum in os.read(read_fd, 16):        # each byte is a signal number
                if signum not in (signal.SIGINT, signal.SIGTERM):
                    continue
                presses += 1
                if presses == 1:
                    timer = threading.Timer(grace_s, hard_exit)
                    timer.daemon = True
                    timer.start()
                else:
                    hard_exit()

    threading.Thread(target=watch, daemon=True, name="signal-watcher").start()
