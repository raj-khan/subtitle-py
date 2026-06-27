# subtitle-py

Live, fully local English subtitles for whatever audio is playing on your laptop.

It listens to your system audio, transcribes the speech in near real time with
Whisper, and shows the captions as a movable overlay you can drag anywhere on
screen. Everything runs on your machine: no audio leaves your computer, no API
keys, no cost.

Works today on Ubuntu (X11 + PulseAudio). Other platforms may come later.

## Why

Sometimes you want subtitles for a video that does not have any: a livestream, a
talk, a recording playing in some other app. This gives you captions for anything
coming out of your speakers.

## Features

- Fully local and free. Uses faster-whisper on the CPU, so it works without a GPU.
- Speech detection (Silero VAD) keeps the screen blank during music and silence
  instead of inventing text.
- Draggable overlay. Pick the font size and color in a small config file.
- Near real time: roughly 2 to 3 seconds behind on a modern laptop CPU.

## Requirements

- Ubuntu with X11 and PulseAudio
- Python 3.10 or newer
- `parec` (from the `pulseaudio-utils` package, usually already installed)

To check that you are on X11 and PulseAudio:

```bash
echo $XDG_SESSION_TYPE          # should say: x11
pactl info | grep "Server Name" # should mention pulseaudio
```

## Install

```bash
git clone https://github.com/raj-khan/subtitle-py.git
cd subtitle-py
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If `python3 -m venv` complains that `ensurepip` is missing, install the venv
package first: `sudo apt install python3-venv`.

The first run downloads the Whisper model (about 75 MB for `base.en`). After that
it is fully offline.

## Usage

Start your video, then run:

```bash
.venv/bin/python main.py
```

Pick your audio output from the menu (usually the one named
"Monitor of ... Speaker + Headphones"). Captions appear near the bottom center of
the screen. Drag them wherever you want. Press Ctrl-C in the terminal to quit.

Want to see the overlay without any audio first? Run:

```bash
.venv/bin/python scripts/step5_overlay.py
```

## Configuration

Edit `config.yaml` and restart. The main settings:

| Key                  | What it does                                          |
|----------------------|-------------------------------------------------------|
| `font_size`          | Caption text size in pixels                           |
| `font_color`         | Text color, hex (e.g. `"#FFFF00"`)                    |
| `background`         | Show a dark box behind the text for readability       |
| `background_opacity` | How solid that box is, 0.0 to 1.0                     |
| `linger_sec`         | How long the last caption stays before it fades       |
| `model`              | `tiny.en` (fastest), `base.en` (default), `small.en`  |

Your dragged position is saved back into `config.yaml` automatically.

## How it works

```
system audio (PulseAudio monitor)
   -> Silero VAD          (only real speech passes through)
   -> faster-whisper      (base.en, CPU, transcribes the speech)
   -> overlay window      (shows the words, drag to move)
```

Whisper is not a streaming model, so to get smooth captions the text is committed
with LocalAgreement: a word is only shown once two passes agree on it, which stops
the captions from flickering and rewriting themselves as you read.

## Limitations

- English only for now.
- Ubuntu / X11 / PulseAudio only. Audio capture and the overlay are kept in
  separate modules so other platforms can be added later.
- Very long speech with no pauses can fall a little behind, because each refresh
  re-transcribes the growing buffer.

## Development

The code is split into small modules under `autotranscript/`, and each build step
has a standalone script under `scripts/` you can run on its own:

| Script                | What it tests                          |
|-----------------------|----------------------------------------|
| `step1_record.py`     | Audio capture and source picker        |
| `step2_transcribe.py` | The Whisper engine on a WAV file        |
| `step3_vad.py`        | The speech-detection gate              |
| `step4_live.py`       | Live captions in the terminal          |
| `step5_overlay.py`    | The overlay with fake captions          |

See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to help.

## License

MIT. See [LICENSE](LICENSE).
