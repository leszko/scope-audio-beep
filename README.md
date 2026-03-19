# Audio Beep Scope Plugin

A [Daydream Scope](https://github.com/daydreamlive/scope) plugin that generates periodic beep tones. Use it as an **audio-only pipeline** for testing audio streaming without a GPU.

## Features

- **Configurable frequency** — 20 Hz to 20 kHz (default 440 Hz)
- **Adjustable timing** — beep duration, interval, and volume
- **Click-free output** — smooth fade-in/fade-out envelope on each beep
- **Wall-clock pacing** — generates samples based on elapsed time to prevent buffer buildup
- **No GPU required** — runs on CPU

## Install

From the Scope app: **Settings > Plugins** > install with:

```
https://github.com/leszko/scope-audio-beep
```

## License

MIT
