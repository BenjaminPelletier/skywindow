# skywindow

A playground for experimenting with projecting a sky sphere through a virtual window
based on the location of a detected human face.  The initial milestone is a simple
preview utility that displays the live feed from a webcam.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/latest/) package manager
- A webcam and the ability to access it from the host operating system

## Getting started

1. Launch the preview window with uv:
   ```bash
   uv run window
   ```
   The first run will resolve and install the declared dependencies.

By default the script opens camera index `0` and names the preview window
`skywindow`.  Override either of those defaults by passing the `--camera-index`
or `--window-name` options after a `--` separator:
```bash
uv run window -- --camera-index 1 --window-name "Secondary camera"
```

Press the `q` key while the preview window has focus to exit the application.
