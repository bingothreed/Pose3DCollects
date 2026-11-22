"""The clip's own audio, streamed next to the viewer.

Rerun 0.31 ships no audio archetype, so the sound can never live inside the recording: the demo decodes
the clip's first audio stream, streams it to the default output device with sounddevice and keeps a
playhead in seconds. The frame loop uses that playhead to hold picture and sound together, and the small
control window (`instanthmr.audio_ui`) turns the sound on and off while the demo runs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

SAMPLE_RATE = 48000


def _open_audio(path: str):
    import av

    container = av.open(path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    return container, stream


def has_audio(path: str) -> bool:
    """True when the file carries at least one audio stream."""
    try:
        container, stream = _open_audio(path)
        ok = stream is not None
        container.close()
        return ok
    except Exception:
        return False


class AudioClip:
    """The decoded audio: mono float32 at `AudioClip.rate`, plus its duration in seconds."""

    def __init__(self, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
        self.samples = np.ascontiguousarray(samples, dtype=np.float32)
        self.rate = int(rate)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.rate


def load_clip(path: str, rate: int = SAMPLE_RATE) -> Optional[AudioClip]:
    """Decode the first audio stream to mono float32 at `rate`; None when there is no audio."""
    import av

    container, stream = _open_audio(path)
    if stream is None:
        container.close()
        return None
    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=rate)
    chunks = []
    try:
        for frame in container.decode(stream):
            for out in resampler.resample(frame) or ():
                chunks.append(out.to_ndarray().astype(np.float32, copy=False).reshape(-1))
        for out in resampler.resample(None) or ():
            chunks.append(out.to_ndarray().astype(np.float32, copy=False).reshape(-1))
    finally:
        container.close()
    if not chunks:
        return None
    clip = AudioClip(np.concatenate(chunks), rate)
    return clip if len(clip.samples) else None


def list_output_devices() -> list[tuple[int, str]]:
    """(index, name) of every device that can play sound."""
    import sounddevice as sd

    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]


def device_name(index: int | None) -> str:
    import sounddevice as sd

    try:
        return str(sd.query_devices(index)["name"])
    except Exception:
        return "system default"


class AudioPlayer:
    """Streams a clip on the default output device. `cursor` is the playhead in seconds.

    The stream stays open for the whole run; muting only silences the callback, so turning the sound back
    on is instant. Seeking is a plain move of the sample position, so it is safe from any thread.
    """

    def __init__(self, clip: AudioClip, enabled: bool = True, volume: float = 1.0,
                 device: int | None = None) -> None:
        self.clip, self.enabled, self.volume = clip, enabled, volume
        self.device = device
        self._pos = 0
        # When set (seconds), the playhead may not run further than `limit + lead`: the picture cannot
        # keep up with real time, so the sound waits for it instead of drifting away.
        self.limit: float | None = None
        self.lead = 0.35
        self.blocks = self.gated = 0
        self._stream = None
        self._open_stream()

    def _open_stream(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = sd.OutputStream(samplerate=self.clip.rate, channels=1, dtype="float32",
                                       blocksize=1024, device=self.device, callback=self._callback)
        self._stream.start()

    def set_device(self, device: int) -> None:
        """Move the stream to another output device, keeping the playhead where it is."""
        self.device = device
        self._open_stream()
        print(f"[audio] output device -> {device_name(device)}")

    # -- audio callback -------------------------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status) -> None:
        samples = self.clip.samples
        self.blocks += 1
        if not self.enabled:
            outdata.fill(0)
            return
        pos = self._pos
        if self.limit is not None and pos >= (self.limit + self.lead) * self.clip.rate:
            self.gated += 1
            outdata.fill(0)
            return
        chunk = samples[pos:pos + frames]
        n = len(chunk)
        if n:
            outdata[:n, 0] = chunk * self.volume
        if n < frames:
            outdata[n:, 0] = 0.0
        self._pos = min(pos + frames, len(samples))

    # -- playhead -------------------------------------------------------------------------------
    @property
    def cursor(self) -> float:
        return self._pos / self.clip.rate

    @property
    def duration(self) -> float:
        return self.clip.duration

    @property
    def at_end(self) -> bool:
        return self._pos >= len(self.clip.samples)

    def seek(self, seconds: float) -> None:
        self._pos = int(min(max(seconds, 0.0), self.duration) * self.clip.rate)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_limit(self, seconds: float | None) -> None:
        """Let the sound run only up to the frame being shown (plus `lead`); None lets it free-run."""
        self.limit = None if seconds is None else float(seconds)

    def gated_ratio(self) -> float:
        """Share of output blocks that were silenced to wait for the picture, 0 when nothing was gated."""
        return self.gated / self.blocks if self.blocks else 0.0

    def stop(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


def open_clip_audio(path: str, rate: int = SAMPLE_RATE, device: int | None = None) -> Optional[AudioPlayer]:
    """Load `path`'s audio and start streaming it; None (with a printed reason) when it cannot play."""
    if not has_audio(path):
        print("[audio] no audio track in this file, nothing to play")
        return None
    try:
        clip = load_clip(path, rate)
    except Exception as e:
        print(f"[audio] could not decode the audio track: {e}")
        return None
    if clip is None:
        print("[audio] the audio track decoded to zero samples, nothing to play")
        return None
    try:
        player = AudioPlayer(clip, device=device)
    except Exception as e:  # no output device, PortAudio missing, ...
        print(f"[audio] could not open the output device: {e}")
        return None
    print(f"[audio] {clip.duration:.1f} s of sound on '{device_name(device)}'; "
          "switch it on/off in the panel")
    return player
