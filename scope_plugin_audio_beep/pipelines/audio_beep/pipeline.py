import math
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline

from .schema import AudioBeepConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

# Generate audio at 48kHz to match WebRTC output rate (avoids resampling)
_AUDIO_RATE = 48000
# Samples generated per pipeline call (~1/30 sec, approximating 30 fps)
_SAMPLES_PER_CALL = 1600


class AudioBeepPipeline(Pipeline):
    """Generates periodic audio beeps with a flashing video frame.

    Each call produces one video frame (black, or white during a beep) and
    a chunk of audio samples. The beep is a pure sine wave at the configured
    frequency, fired at a configurable interval.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return AudioBeepConfig

    def __init__(self, height: int = 512, width: int = 512, **kwargs):
        self.height = height
        self.width = width
        self._sample_offset = 0

    def __call__(self, **kwargs) -> dict:
        frequency = kwargs.get("frequency", 440.0)
        beep_duration = kwargs.get("beep_duration", 0.1)
        beep_interval = kwargs.get("beep_interval", 1.0)
        volume = kwargs.get("volume", 0.5)

        # Generate audio samples for this chunk
        samples = []
        for i in range(_SAMPLES_PER_CALL):
            t = (self._sample_offset + i) / _AUDIO_RATE
            t_in_cycle = t % beep_interval
            if t_in_cycle < beep_duration:
                sample = volume * math.sin(2.0 * math.pi * frequency * t)
            else:
                sample = 0.0
            samples.append(sample)

        self._sample_offset += _SAMPLES_PER_CALL

        audio = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)  # [1, S]

        # Video: white frame during beep, black otherwise
        t_now = self._sample_offset / _AUDIO_RATE
        in_beep = (t_now % beep_interval) < beep_duration
        brightness = 1.0 if in_beep else 0.0
        frame = torch.full((1, self.height, self.width, 3), brightness, dtype=torch.float32)

        return {
            "video": frame,
            "audio": audio,
            "audio_sample_rate": _AUDIO_RATE,
        }
