# skywindow

A playground for experimenting with projecting a sky sphere through a virtual window
based on the location of a detected human face.  The initial milestone is a simple
preview utility that displays the live feed from a webcam.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/latest/) package manager
- A webcam and the ability to access it from the host operating system

## Getting started

1. Create a virtual environment and install dependencies with uv:
   ```bash
   uv venv
   uv pip install -r pyproject.toml
   ```
   Alternatively, you can activate the environment and rely on uv's resolver:
   ```bash
   source .venv/bin/activate
   uv pip install opencv-python
   ```
2. Launch the preview window:
   ```bash
   python window.py
   ```

By default the script opens camera index `0` and names the preview window
`skywindow`.  Override either of those defaults by passing the `--camera-index`
or `--window-name` options.

Press the `q` key while the preview window has focus to exit the application.
