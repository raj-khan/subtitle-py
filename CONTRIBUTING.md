# Contributing

Thanks for taking a look. This is a small project, so the process is simple.

## Getting set up

```bash
git clone https://github.com/raj-khan/subtitle-py.git
cd subtitle-py
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Each part of the pipeline has its own script under `scripts/`, so you can work on
one piece without running the whole app. Start there.

## Ways to help

- Try it on your machine and open an issue if something breaks. Include your
  Ubuntu version, `echo $XDG_SESSION_TYPE`, and `pactl info | grep "Server Name"`.
- Improve accuracy or latency tuning (VAD thresholds, refresh timing).
- Add support for another platform. Audio capture lives in
  `autotranscript/capture.py` and the overlay in `autotranscript/overlay.py`,
  both written so a Windows or macOS version can sit beside the Linux one.

## Pull requests

- Keep each change small and focused on one thing.
- Match the style of the code around you.
- Explain what you changed and how you tested it.
- Test your change before opening the PR. If you touched the pipeline, say which
  `scripts/stepN_*.py` you ran and what you saw.

## Reporting bugs

Open an issue with what you expected, what happened, and the steps to reproduce.
Logs or a short clip of the behavior help a lot.
