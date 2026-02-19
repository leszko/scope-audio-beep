import math
import time
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline

from .schema import AudioBeepConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

# Generate audio at 48kHz to match WebRTC output rate (avoids resampling)
_AUDIO_RATE = 48000
# Default sample count for the very first call (~33ms of audio)
_INITIAL_SAMPLES = 1600
# Hard cap: never generate more than 1 second of audio in a single call.
# Prevents a large burst after long pauses.
_MAX_SAMPLES_PER_CALL = _AUDIO_RATE


class AudioBeepPipeline(Pipeline):
    """Generates periodic audio beeps with a flashing video frame.

    Each call produces one video frame (black, or white during a beep) and
    a chunk of audio samples. The beep is a pure sine wave at the configured
    frequency, fired at a configurable interval.

    Audio is generated based on elapsed wall-clock time so the pipeline
    produces audio at approximately real-time regardless of how fast the
    pipeline loop runs. Without this, the audio buffer would grow unboundedly
    (the pipeline has no video input to pace it), and UI parameter changes
    would only be heard after the entire backlog drains.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return AudioBeepConfig

    def __init__(self, height: int = 512, width: int = 512, **kwargs):
        self.height = height
        self.width = width
        self._sample_offset = 0
        self._last_call_time: float | None = None

    def __call__(self, **kwargs) -> dict:
        frequency = kwargs.get("frequency", 440.0)
        beep_duration = kwargs.get("beep_duration", 0.1)
        beep_interval = kwargs.get("beep_interval", 1.0)
        volume = kwargs.get("volume", 0.5)

        # Determine how many audio samples to generate for this call.
        # We use elapsed wall-clock time so audio is produced at real-time rate,
        # keeping the downstream buffer small and parameter changes audible quickly.
        now = time.monotonic()
        if self._last_call_time is None:
            num_samples = _INITIAL_SAMPLES
        else:
            elapsed = now - self._last_call_time
            num_samples = int(elapsed * _AUDIO_RATE)
        self._last_call_time = now
        num_samples = max(1, min(num_samples, _MAX_SAMPLES_PER_CALL))

        # Vectorised sine-wave generation
        t_start = self._sample_offset / _AUDIO_RATE
        t = torch.linspace(t_start, t_start + num_samples / _AUDIO_RATE, num_samples)
        t_in_cycle = t % beep_interval
        in_beep_mask = t_in_cycle < beep_duration
        audio = (
            torch.where(
                in_beep_mask,
                volume * torch.sin(2.0 * math.pi * frequency * t),
                torch.zeros(num_samples),
            )
            .float()
            .unsqueeze(0)
        )  # [1, S]

        self._sample_offset += num_samples

        # Video: white frame during beep, black otherwise
        t_now = self._sample_offset / _AUDIO_RATE
        in_beep = (t_now % beep_interval) < beep_duration
        brightness = 1.0 if in_beep else 0.0
        frame = torch.full(
            (1, self.height, self.width, 3), brightness, dtype=torch.float32
        )

        return {
            "video": frame,
            "audio": audio,
            "audio_sample_rate": _AUDIO_RATE,
        }
