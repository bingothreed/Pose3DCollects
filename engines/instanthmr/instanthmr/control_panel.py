"""A compact floating switch bar docked inside the Rerun Viewer window.

Rerun 0.31 gives Python neither audio playback nor widgets -- custom panels are Rust-only (`eframe` +
`re_viewer`) -- and its viewer never reports a click back to us. So the switches live in this panel: the
audio ones drive `AudioPlayer` directly and the 3D ones re-send the blueprint live.

To look like part of the page rather than a second window, the panel is frameless, always-on-top and
keeps itself glued to the top-right corner of the `Rerun Viewer` window (`_dock`), following it while it
is moved or resized. Drag it anywhere to undock; drag it back over the viewer to keep it there.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import time
from typing import Callable, Optional

from instanthmr.audio import AudioPlayer, list_output_devices, device_name
from instanthmr.visualizer import RerunVisualizer

VIEWER_TITLE = "Rerun Viewer"


def _find_viewer_window() -> Optional[int]:
    """The hwnd of the (first, visible) Rerun Viewer top-level window, or None."""
    if not hasattr(ctypes, "windll"):
        return None
    found: list[int] = []

    def visit(hwnd, _lparam):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True
        n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if not n:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 1)
        if VIEWER_TITLE.lower() in buf.value.lower():
            found.append(hwnd)
            return False
        return True

    ctypes.windll.user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(visit), 0)
    return found[0] if found else None


def _client_rect_on_screen(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """(left, top, width, height) of `hwnd`'s client area in screen pixels."""
    if not hwnd:
        return None
    rect = wt.RECT()
    point = wt.POINT(0, 0)
    if not ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point)):
        return None
    return point.x, point.y, rect.right - rect.left, rect.bottom - rect.top


def _window_rect(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    if not hwnd:
        return None
    rect = wt.RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


class ControlPanel:
    """Sound on/off + strict A/V sync + volume, and what the 3D view draws."""

    def __init__(self, player: Optional[AudioPlayer] = None, viewer: Optional[RerunVisualizer] = None,
                 clock: Callable[[], float] = lambda: 0.0, has_audio: bool = True,
                 dock: bool = True, video_path: Optional[str] = None) -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self._tk = tk
        self.player, self.viewer = player, viewer
        self.video_path = video_path
        self._clock = clock
        self.closed = False
        # True until the frame loop is over; replay only makes sense on a finished recording.
        self.processing = True
        self._replay_anchor: Optional[float] = None
        self._replay_after: Optional[str] = None
        self._viewer_hwnd: Optional[int] = None
        self._docked = False
        self._dock_pending = False
        self._dragged = False
        self._polls = 0
        self._drag_origin = (0, 0)

        self.root = tk.Tk()
        self.root.title("InstantHMR 控制面板")
        try:
            tiny = tkfont.Font(family="Microsoft YaHei UI", size=9)
        except Exception:
            tiny = None
        body = tk.Frame(self.root, padx=8, pady=5)
        body.pack()
        self._body = body

        self.sound = tk.BooleanVar(value=bool(player and player.enabled))
        self.sync = tk.BooleanVar(value=True)
        self.volume = tk.DoubleVar(value=player.volume * 100.0 if player else 100.0)

        opts = {"font": tiny} if tiny else {}
        row_a = tk.Frame(body)
        row_a.grid(row=0, column=0, sticky="w")
        if player is None:
            tk.Label(row_a, text="该文件没有音轨" if not has_audio else "未启用原声", **opts).pack(side="left")
        else:
            tk.Checkbutton(row_a, text="原声同步播放", variable=self.sound, command=self._apply_sound,
                           **opts).pack(side="left")
            tk.Checkbutton(row_a, text="严格对齐", variable=self.sync, **opts).pack(side="left", padx=(6, 0))
            tk.Scale(row_a, from_=0, to=100, orient="horizontal", length=90, variable=self.volume,
                     showvalue=False, command=lambda _v: self._apply_volume()).pack(side="left", padx=(6, 0))

            row_dev = tk.Frame(body)
            row_dev.grid(row=1, column=0, sticky="w")
            tk.Label(row_dev, text="输出", **opts).pack(side="left")
            # The system default is often a headset while people listen on the monitor speakers, which
            # looks exactly like "no sound" -- so make the device switchable instead of guessing.
            self._devices = list_output_devices()
            current = player.device if player.device is not None else None
            label = f"{current}: {device_name(current)}" if current is not None else "(系统默认)"
            if self._devices and current is not None and current not in [i for i, _ in self._devices]:
                label = "(系统默认)"
            self.device_label = tk.StringVar(value=label)
            menu = tk.OptionMenu(row_dev, self.device_label, *[f"{i}: {n}" for i, n in self._devices],
                                 command=self._switch_device)
            if self._devices:
                if tiny:
                    menu.configure(font=tiny)
                menu.pack(side="left", padx=(4, 0))
            else:
                tk.Label(row_dev, text=device_name(player.device), **opts).pack(side="left")

        if viewer is not None:
            row_b = tk.Frame(body)
            row_b.grid(row=2, column=0, sticky="w")
            self.camera_in_3d = tk.BooleanVar(value=bool(viewer.camera_in_3d))
            tk.Checkbutton(row_b, text="3D 相机/图像平面", variable=self.camera_in_3d,
                           command=lambda: viewer.update_layout(camera_in_3d=self.camera_in_3d.get()),
                           **opts).pack(side="left")
            self.floor = tk.BooleanVar(value=bool(viewer.floor))
            tk.Checkbutton(row_b, text="3D 地面网格", variable=self.floor,
                           command=lambda: viewer.update_layout(floor=self.floor.get()),
                           **opts).pack(side="left", padx=(6, 0))
            self.viewer_grid = tk.BooleanVar(value=bool(viewer.viewer_grid))
            tk.Checkbutton(row_b, text="3D 内置网格", variable=self.viewer_grid,
                           command=lambda: viewer.update_layout(viewer_grid=self.viewer_grid.get()),
                           **opts).pack(side="left", padx=(6, 0))

        row_c = tk.Frame(body)
        row_c.grid(row=3, column=0, sticky="we")
        self.status = tk.Label(row_c, text="", anchor="w", font=("Consolas", 8))
        self.status.pack(side="left")
        if player is not None:
            tk.Button(row_c, text="▶ 从头播放", bd=1, padx=5, pady=0, command=self.replay,
                      **opts).pack(side="right", padx=(0, 4))
        if self.video_path:
            # The only way to get picture and sound in one playback: the exported mp4, opened in the
            # system player. The Rerun viewer itself cannot play audio.
            tk.Button(row_c, text="▶ 带声音的视频", bd=1, padx=5, pady=0, command=self.open_video,
                      **opts).pack(side="right", padx=(0, 4))
        tk.Button(row_c, text="✕", bd=0, padx=5, pady=0, command=self.close,
                  **opts).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh_status()
        self.root.update_idletasks()
        self._dock_target = dock and viewer is not None
        self._apply_dock()
        # Drag by the padding, not by the controls, so clickging a checkbox or the slider still works.
        self._drag_handles = [body, row_a, row_c, self.status]
        if viewer is not None:
            self._drag_handles.append(row_b)
        if player is not None:
            self._drag_handles.append(row_dev)
        for widget in self._drag_handles:
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
        self.root.update()

    # -- docking ---------------------------------------------------------------------------------
    def _apply_dock(self) -> None:
        """Frameless, on top and glued to the viewer's top-right corner."""
        if not self._dock_target:
            try:
                self.root.attributes("-topmost", True)
            except Exception:
                pass
            return
        self._viewer_hwnd = _find_viewer_window()
        if self._viewer_hwnd is None:
            # The viewer is spawned as a separate process and its window may take a moment to show up;
            # keep the normal window meanwhile and retry from `poll()`.
            self._dock_pending = True
            print("[panel] waiting for the Rerun Viewer window to dock into")
            try:
                self.root.attributes("-topmost", True)
            except Exception:
                pass
            return
        try:
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
        except Exception:
            self._dock_target = False
            return
        self._docked = True
        self._dock_pending = False
        print("[panel] docked to the top-right corner of the Rerun Viewer window")
        try:
            self.root.deiconify()
            self.root.lift()
        except Exception:
            pass
        self._snap()

    def _snap(self, margin: int = 10) -> None:
        client = _client_rect_on_screen(self._viewer_hwnd)
        if client is None:
            return
        self.root.update_idletasks()
        w, h = self._panel_size()
        x = client[0] + client[2] - w - margin
        y = client[1] + margin
        self._place(x, y)

    def _panel_size(self) -> tuple[int, int]:
        """The size the packed widgets ask for; `winfo_width` is stale right after `overrideredirect`."""
        return (max(self.root.winfo_reqwidth(), 1), max(self.root.winfo_reqheight(), 1))

    def _place(self, x: int, y: int) -> None:
        # Tk's `+x+y` geometry is in screen pixels here (checked against GetWindowRect), so no DPI maths.
        # The size is set explicitly: switching to a frameless window otherwise collapses the layout.
        w, h = self._panel_size()
        self.root.geometry(f"{w}x{h}+{int(round(x))}+{int(round(y))}")

    def _follow_viewer(self) -> None:
        if not self._docked or self._dragged or not self._viewer_hwnd:
            return
        if not ctypes.windll.user32.IsWindow(self._viewer_hwnd):
            self._docked = False
            return
        client = _client_rect_on_screen(self._viewer_hwnd)
        if client is None:
            return
        own = _window_rect(self.root.winfo_id())
        if own is None:
            return
        want_x = client[0] + client[2] - own[2] - 10
        want_y = client[1] + 10
        if abs(own[0] - want_x) > 2 or abs(own[1] - want_y) > 2:
            self._place(want_x, want_y)

    def _drag_start(self, event) -> None:
        self._drag_origin = (event.x_root, event.y_root)
        self._dragged = True

    def _drag_move(self, event) -> None:
        dx = event.x_root - self._drag_origin[0]
        dy = event.y_root - self._drag_origin[1]
        self._drag_origin = (event.x_root, event.y_root)
        self._place(self.root.winfo_x() + dx, self.root.winfo_y() + dy)

    # -- audio -----------------------------------------------------------------------------------
    def _apply_sound(self) -> None:
        if self.player is None:
            return
        on = bool(self.sound.get())
        self.player.set_enabled(on)
        if on:
            self.player.seek(self._sound_position())
            print(f"[audio] sound on, realigned to {self.player.cursor:.1f} s")
        else:
            print("[audio] sound off")

    def _sound_position(self) -> float:
        """Where the sound should resume: the clip being replayed, or the frame being processed."""
        if self._replay_anchor is not None and self.player is not None:
            return min(time.perf_counter() - self._replay_anchor, self.player.duration)
        return self._clock()

    def _apply_volume(self) -> None:
        if self.player is not None:
            self.player.volume = float(self.volume.get()) / 100.0

    def _switch_device(self, label: str) -> None:
        if self.player is None:
            return
        try:
            index = int(str(label).split(":", 1)[0])
        except ValueError:
            return
        self.player.set_device(index)

    def open_video(self) -> None:
        if not self.video_path:
            return
        try:
            os.startfile(self.video_path)  # noqa: S606 - opening our own export in the user's player
        except Exception as e:
            print(f"[panel] could not open {self.video_path}: {e}")

    def replay(self) -> None:
        """Replay the finished recording from the start, with the clip's sound in step.

        The viewer cannot tell us when the user presses play, but it does apply the time panel's
        play state from the blueprint. Asking it to follow first puts the cursor on the last frame;
        asking it to play then takes the viewer's "at the end, start from the beginning" path, so
        the picture restarts on the first frame at the same moment the sound restarts.
        """
        if self.processing:
            print("[panel] the frames are still being processed; replay once that is done")
            return
        if self._replay_after is not None:
            return  # already waiting for the viewer to jump to the end; ignore the second click
        if self.viewer is not None and not self.viewer_alive():
            self.viewer = None
        if self.viewer is not None:
            self.viewer.set_time_panel(play_state="Following")
            # One blueprint message per viewer frame: the viewer only reads the latest one.
            self._replay_after = self.root.after(140, self._start_replay)
        else:
            self._start_replay()

    def _start_replay(self) -> None:
        self._replay_after = None
        if self.closed:
            return
        if self.player is not None:
            self.player.seek(0.0)
            self.player.set_limit(None)
            self.player.set_enabled(True)
            self.sound.set(True)
            self._replay_anchor = time.perf_counter()
        if self.viewer is not None:
            self.viewer.set_time_panel(play_state="Playing")
            print("[panel] replaying from the start: the viewer plays the timeline in real time and "
                  "the clip's sound runs with it")
        elif self.player is not None:
            print("[audio] replaying from the start; press play in the viewer at the same moment to line up")

    def viewer_alive(self) -> bool:
        """False once the Rerun Viewer window has been closed."""
        if not self._viewer_hwnd or not hasattr(ctypes, "windll"):
            return True
        return bool(ctypes.windll.user32.IsWindow(self._viewer_hwnd))

    # -- per frame -------------------------------------------------------------------------------
    def poll(self) -> None:
        """Pump the window's events, keep it docked and refresh the readout."""
        if self.closed:
            return
        try:
            self.root.update()
        except Exception:
            self.closed = True
            return
        self._polls += 1
        if self._polls % 20 == 0:
            if self._dock_target and not self._docked and self._dock_pending:
                self._apply_dock()
            else:
                self._follow_viewer()
        try:
            self._refresh_status()
        except Exception:
            # Tk raises "invalid command name" once a widget is gone; the run must not die with it.
            self.closed = True

    def _refresh_status(self) -> None:
        if self.player is None:
            self.status.configure(text="无音轨")
            return
        state = "播放中" if self.player.enabled else "已静音"
        if self.player.enabled and self.player.at_end:
            state = "播放完毕"
        self.status.configure(text=f"{self.player.cursor:4.1f}/{self.player.duration:4.1f}s {state}")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.player is not None:
            self.player.stop()
        try:
            self.root.destroy()
        except Exception:
            pass
