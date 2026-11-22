"""Write the demo's frames -- the 2D view with boxes, skeleton and timings -- to an mp4 with the clip's
own audio, so the recording can be watched (and heard) in an ordinary player.

The Rerun viewer has no audio support in 0.31, so this is the only way to get picture and sound in one
file: H.264 video encoded with PyAV plus the clip's audio track, transcoded to AAC.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from instanthmr.audio import AudioClip


class OverlayVideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int], audio: Optional[AudioClip] = None,
                 bitrate: int = 6_000_000) -> None:
        import av

        self.path = path
        self.fps = max(float(fps), 1.0)
        self.size = size
        self.audio = audio
        self._frame = 0
        self.container = av.open(path, "w")
        self.video = self.container.add_stream("libx264", rate=int(round(self.fps)))
        self.video.width, self.video.height = size
        self.video.pix_fmt = "yuv420p"
        try:
            self.video.bit_rate = bitrate
        except Exception:
            pass
        self.audio_stream = None
        if audio is not None:
            self.audio_stream = self.container.add_stream("aac", rate=audio.rate)
            self.audio_stream.layout = "mono"

    def write(self, rgb: np.ndarray) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts = self._frame
        self._frame += 1
        for packet in self.video.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        import av

        for packet in self.video.encode(None):
            self.container.mux(packet)
        if self.audio_stream is not None and self.audio is not None:
            samples, rate = self.audio.samples, self.audio.rate
            chunk = rate // 10
            for start in range(0, len(samples), chunk):
                block = samples[start:start + chunk].reshape(1, -1).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="mono")
                frame.sample_rate = rate
                frame.pts = start
                for packet in self.audio_stream.encode(frame):
                    self.container.mux(packet)
            for packet in self.audio_stream.encode(None):
                self.container.mux(packet)
        self.container.close()
