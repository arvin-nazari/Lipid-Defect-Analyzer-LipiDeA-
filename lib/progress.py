"""
Simple terminal progress reporting.

One progress bar at a time, driven by prep.in's ``verbose`` key.

verbose: on   Nothing in this module does anything. 
verbose: off  The run's file descriptor 1 is pointed at results/run.log, and
              one progress bar per pipeline step is drawn on the real stderr
              instead.

"""

from contextlib import contextmanager
from pathlib import Path
import os
import sys
import time
import warnings

# ==================================================
# Module state
# ==================================================

BAR_WIDTH = 28
REDRAW_INTERVAL = 0.1           # seconds; minimum gap between TTY redraws

_verbose = True
_log_file = None
_saved_stdout_fd = None         # dup of the real fd 1, restored by close()
_saved_showwarning = None

_bar_open = False
_bar_label = ""
_bar_total = None               # None = indeterminate
_bar_done = 0
_bar_t0 = 0.0
_bar_last_len = 0
_bar_last_draw = 0.0
_bar_tty = False
_bar_width = 80                 # terminal columns, sampled per bar


# ==================================================
# Bar rendering
# ==================================================
def _terminal_width():
    """Columns available on the real stderr, with a conservative fallback.

    os.get_terminal_size looks at fd 1 by default, which in quiet mode is the
    run log and has no size at all, so this asks fd 2 explicitly. Falls back
    to 80 for a pipe or a terminal that will not report.
    """
    try:
        return max(20, os.get_terminal_size(sys.__stderr__.fileno()).columns)
    except Exception:
        return 80


        
def _bar_line(final=False):
    """Build the current bar text: a filled bar with a fraction, or a plain status if no total is known.
    """
    elapsed = time.perf_counter() - _bar_t0

    if _bar_total:
        done = min(_bar_done, _bar_total)
        frac = done / _bar_total
        filled = int(round(BAR_WIDTH * frac))
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        return (
            f"{_bar_label}  {bar} {done}/{_bar_total} "
            f"{100.0 * frac:3.0f}%  {elapsed:7.1f}s"
        )

    return f"{_bar_label}  {'done' if final else 'working'}  {elapsed:7.1f}s"


def _draw_bar(final=False, force=False):
    """Redraw the bar in place on the real stderr. The only function here that writes a bar.

    """
    global _bar_last_len, _bar_last_draw

    now = time.perf_counter()

    if not final:
        if not _bar_tty:
            return
        if not force and (now - _bar_last_draw) < REDRAW_INTERVAL:
            return

    _bar_last_draw = now
    line = _bar_line(final=final)

    if not _bar_tty:
        sys.__stderr__.write(line + "\n")
        sys.__stderr__.flush()
        return

    pad = " " * max(0, _bar_last_len - len(line))
    _bar_last_len = len(line)
    sys.__stderr__.write("\r" + line + pad + ("\n" if final else ""))
    sys.__stderr__.flush()


def _clear_bar_line():
    """Blank the bar's line so another writer can use the terminal cleanly."""
    global _bar_last_len

    if _bar_tty and _bar_last_len:
        sys.__stderr__.write("\r" + " " * _bar_last_len + "\r")
        sys.__stderr__.flush()
        _bar_last_len = 0


# ==================================================
# Output mode
# ==================================================

def _log_warning(message, category, filename, lineno, file=None, line=None):
    """Write a Python warning to the run log instead of stderr.

    """
    try:
        sys.stdout.write(warnings.formatwarning(message, category, filename, lineno, line))
        sys.stdout.flush()
    except Exception:
        # A warning that cannot be logged must never take the run down.
        pass


def configure(verbose, log_path=None, capture_stdout=True):
    """Set the output mode for the whole process. Called once, from main().
    """
    global _verbose, _log_file, _saved_stdout_fd, _saved_showwarning

    _verbose = bool(verbose)
    if _verbose:
        return

    _saved_showwarning = warnings.showwarning
    warnings.showwarning = _log_warning

    if not capture_stdout or log_path is None:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = path.open("a", encoding="utf-8", buffering=1)
    _log_file.write(f"\n===== run started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    _log_file.flush()

    sys.stdout.flush()
    _saved_stdout_fd = os.dup(1)
    os.dup2(_log_file.fileno(), 1)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def close():
    """Close any open bar, restore fd 1 and the warning handler, close the log. Safe to call twice."""
    global _log_file, _saved_stdout_fd, _saved_showwarning

    _stop_bar()

    if _saved_showwarning is not None:
        warnings.showwarning = _saved_showwarning
        _saved_showwarning = None

    if _saved_stdout_fd is not None:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.dup2(_saved_stdout_fd, 1)
        os.close(_saved_stdout_fd)
        _saved_stdout_fd = None

    if _log_file is not None:
        _log_file.close()
        _log_file = None


def is_verbose():
    """True when the old print-everything behavior is in force."""
    return _verbose


def log_stream():
    """The open run-log file, or None in verbose mode.

    """
    return _log_file


# ==================================================
# Bar lifecycle
# ==================================================

def _start_bar(label, total):
    """Initialize the bar globals for one step and draw its opening line."""
    global _bar_open, _bar_label, _bar_total, _bar_done, _bar_t0
    global _bar_last_len, _bar_last_draw, _bar_tty

    _bar_open = True
    _bar_label = label
    _bar_total = None if total is None else max(0, int(total))
    _bar_done = 0
    _bar_t0 = time.perf_counter()
    _bar_last_len = 0
    _bar_last_draw = 0.0
    _bar_tty = bool(getattr(sys.__stderr__, "isatty", lambda: False)())

    if _bar_tty:
        _draw_bar(force=True)


def _stop_bar():
    """Write the bar's one and only finished line. Safe with no bar open, and safe to call twice."""
    global _bar_open

    if not _bar_open:
        return

    _bar_open = False
    _draw_bar(final=True)


# ==================================================
# Step-facing API
# ==================================================

def notice(message):
    """Print one line that stays visible in both modes, without breaking an open bar."""
    if _verbose:
        print(message)
        return

    if _bar_open:
        _clear_bar_line()

    sys.__stderr__.write(str(message).lstrip("\n") + "\n")
    sys.__stderr__.flush()

    if _bar_open:
        _draw_bar(force=True)


def set_total(total):
    """Give the open bar its job count, turning the opening line into a real bar.
    """
    global _bar_total

    if not _bar_open:
        return

    _bar_total = None if total is None else max(0, int(total))
    _draw_bar(force=True)


def tick(n=1):
    """Mark n jobs complete and redraw. No-op in verbose mode or with no bar open.

    """
    global _bar_done

    if not _bar_open:
        return

    _bar_done += int(n)
    _draw_bar()


@contextmanager
def step(step_number, title, total=None):
    """Open a progress bar for one pipeline step.

    """
    if _verbose:
        yield
        return

    _start_bar(f"Step {step_number}: {title}", total)
    try:
        yield
    finally:
        _stop_bar()