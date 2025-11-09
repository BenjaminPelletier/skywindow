"""Real-time webcam viewer using OpenCV.

This module exposes a ``main`` entry point that opens a window displaying frames
from the default webcam.  Press ``q`` to quit the application.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
from typing import Optional

import cv2


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the viewer."""
    parser = argparse.ArgumentParser(
        description=(
            "Display real-time video from a webcam. Press 'q' while the "
            "preview window is focused to exit."
        )
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help=(
            "Index of the camera to open. Defaults to 0, which is typically the "
            "system's primary webcam."
        ),
    )
    parser.add_argument(
        "--window-name",
        default="skywindow",
        help="Title to use for the preview window.",
    )
    return parser.parse_args()


def open_capture(camera_index: int) -> cv2.VideoCapture:
    """Return an opened ``cv2.VideoCapture`` for the provided camera index."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "Unable to open camera index {index}. Ensure a webcam is connected."
            .format(index=camera_index)
        )
    return capture


def display_feed(capture: cv2.VideoCapture, window_name: str) -> None:
    """Read frames from ``capture`` and display them until the user quits."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Failed to read frame from the webcam.")

            cv2.imshow(window_name, frame)

            # ``waitKey`` returns -1 when no key is pressed.  Mask to a byte value
            # because OpenCV sometimes returns 32-bit integers on certain platforms.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Entry point used by ``python window.py``."""
    parsed_args = args or parse_args()
    capture = open_capture(parsed_args.camera_index)
    display_feed(capture, parsed_args.window_name)


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
