"""
Тонкий indicator у курсора мыши.

Маленькая точка, которая следует за системным курсором с лёгким смещением
(~14px вниз-вправо) и пульсирует. Показывается только пока идёт запись —
видно "идёт ли запись" не отрываясь от поля ввода.

Tk запускается в собственном daemon-thread'е (Tk не любит вызовы из чужих
потоков). Внешний код шлёт команды через queue.Queue, GUI вычитывает их
через Tk.after(). Зависимости: только stdlib.

⚠️ macOS: НЕ ПОДДЕРЖИВАЕТСЯ. macOS Tk требует чтобы Tk() был на main thread
(NSApplication привязан к main thread'у). Создание Tk в фоне даёт:
    NSInvalidArgumentException '-[NSApplication macOSVersion]: unrecognized selector'
с последующим крашем процесса. С KeepAlive=true в LaunchAgent — бесконечный
краш-цикл. Поэтому на macOS CursorIndicator.start() — no-op.
Пользователи Mac получают visual feedback через tray icon (TrayIcon).

Windows-only нюанс: позиция курсора берётся через GetCursorPos из user32.
На Linux fallback на pyautogui.position(), если pyautogui установлен.

Использование:
    from scripts.cursor_indicator import CursorIndicator

    ind = CursorIndicator(color="#ef4444")
    ind.start()
    # ... начали запись ...
    ind.show()
    # ... закончили ...
    ind.hide()
    # ... выход ...
    ind.stop()
"""

from __future__ import annotations

import logging
import math
import platform
import queue
import threading
import time
from typing import Optional, Tuple


SIZE = 6                     # window size (square)
DOT_RADIUS = 1.5             # tiny solid red dot
OFFSET_X = 10                # cursor → dot offset (right of cursor)
OFFSET_Y = 6                 # ~middle of arrow cursor height
TRANSPARENT_BG = "#ff00ff"   # transparent-key colour (magenta) on Windows
DEFAULT_COLOR = "#ef4444"    # red-500
BLINK_PERIOD_S = 0.6         # one full on/off cycle (caret-style blink)
TICK_MS = 33                 # ~30 fps
POLL_MS = 15


def _get_cursor_pos_win32() -> Optional[Tuple[int, int]]:
    """Return (x, y) of system cursor in screen coords, or None."""
    try:
        import ctypes
        from ctypes import wintypes

        class _POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        pt = _POINT()
        ok = ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        if not ok:
            return None
        return int(pt.x), int(pt.y)
    except Exception:
        return None


def _get_cursor_pos_fallback() -> Optional[Tuple[int, int]]:
    try:
        import pyautogui
        x, y = pyautogui.position()
        return int(x), int(y)
    except Exception:
        return None


def _get_cursor_pos() -> Optional[Tuple[int, int]]:
    if platform.system() == "Windows":
        pos = _get_cursor_pos_win32()
        if pos is not None:
            return pos
    return _get_cursor_pos_fallback()


class CursorIndicator:
    def __init__(self, color: str = DEFAULT_COLOR):
        self.color = color
        self._cmd_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cursor-indicator")
        # macOS не поддерживает Tk в non-main thread (NSInvalidArgumentException).
        # Класс молча превращается в no-op чтобы caller-код можно было оставить
        # без условий — show()/hide()/stop() просто ничего не делают.
        self._disabled_on_mac = platform.system() == "Darwin"

    # ─── Public API ─────────────────────────────────────────────────────

    def start(self) -> None:
        if self._disabled_on_mac:
            logging.info(
                "cursor_indicator: skipped on macOS (Tk thread-safety issue). "
                "Visual feedback available via show_tray=true."
            )
            self._ready.set()
            return
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def stop(self) -> None:
        if self._disabled_on_mac:
            return
        self._cmd_queue.put(("__quit__", None))
        self._stopped.wait(timeout=1.0)

    def show(self) -> None:
        if self._disabled_on_mac:
            return
        self._cmd_queue.put(("show", None))

    def hide(self) -> None:
        if self._disabled_on_mac:
            return
        self._cmd_queue.put(("hide", None))

    # ─── Internals (Tk thread only) ─────────────────────────────────────

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as e:
            logging.error(f"cursor_indicator: tkinter unavailable: {e}")
            self._ready.set()
            return

        try:
            self._tk = tk.Tk()
        except Exception as e:
            logging.error(f"cursor_indicator: Tk() failed: {e}")
            self._ready.set()
            return

        self._tk.withdraw()  # hide root

        self._win = tk.Toplevel(self._tk)
        self._win.overrideredirect(True)
        self._win.wm_attributes("-topmost", True)
        try:
            # Magenta background becomes fully transparent on Windows.
            self._win.wm_attributes("-transparentcolor", TRANSPARENT_BG)
        except Exception:
            pass
        # Disable input — clicks pass through to whatever is below.
        try:
            self._win.wm_attributes("-disabled", True)
        except Exception:
            pass
        self._win.geometry(f"{SIZE}x{SIZE}+0+0")
        self._win.withdraw()

        self._canvas = tk.Canvas(
            self._win, width=SIZE, height=SIZE,
            bg=TRANSPARENT_BG, highlightthickness=0, bd=0,
        )
        self._canvas.pack()

        self._is_visible = False
        self._target_visible = False
        self._anim_phase = 0.0

        self._tk.after(POLL_MS, self._poll_commands)
        self._tk.after(TICK_MS, self._tick)

        self._ready.set()

        try:
            self._tk.mainloop()
        except Exception as e:
            logging.error(f"cursor_indicator: mainloop crashed: {e}")
        finally:
            self._stopped.set()

    def _poll_commands(self) -> None:
        try:
            while True:
                cmd, _ = self._cmd_queue.get_nowait()
                if cmd == "__quit__":
                    self._tk.quit()
                    return
                elif cmd == "show":
                    self._target_visible = True
                elif cmd == "hide":
                    self._target_visible = False
        except queue.Empty:
            pass
        try:
            self._tk.after(POLL_MS, self._poll_commands)
        except Exception:
            pass

    def _tick(self) -> None:
        # Toggle visibility per command
        if self._target_visible and not self._is_visible:
            self._win.deiconify()
            self._is_visible = True
        elif not self._target_visible and self._is_visible:
            self._win.withdraw()
            self._is_visible = False

        if self._is_visible:
            self._anim_phase += TICK_MS / 1000.0

            # Follow cursor
            pos = _get_cursor_pos()
            if pos is not None:
                cx, cy = pos
                self._win.geometry(f"{SIZE}x{SIZE}+{cx + OFFSET_X}+{cy + OFFSET_Y}")

            # Pulse the dot radius
            self._draw()

        try:
            self._tk.after(TICK_MS, self._tick)
        except Exception:
            pass

    def _draw(self) -> None:
        c = self._canvas
        c.delete("all")

        # Binary blink: dot is visible during the first half of the period,
        # hidden during the second. Mimics a text-caret blink rather than a
        # smooth breath.
        phase = (self._anim_phase % BLINK_PERIOD_S) / BLINK_PERIOD_S
        if phase >= 0.5:
            return  # off — leave canvas empty (transparent)

        cx = SIZE / 2
        cy = SIZE / 2
        r = DOT_RADIUS
        c.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=self.color, outline="",
        )
