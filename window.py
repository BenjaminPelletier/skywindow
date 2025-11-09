"""Real-time webcam viewer using OpenCV.

This module exposes a ``main`` entry point that opens a window displaying frames
from the default webcam.  Press ``q`` to quit the application.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
import time
from typing import Optional, Sequence, Tuple

import cv2

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Tuning constants for the mid-point smoothing filter.  ``ALPHA`` controls how much of
# the raw measurement is blended into the filtered position every frame, ``BETA``
# controls responsiveness to changes in velocity, and ``GAMMA`` determines how
# aggressively acceleration corrections are applied.  ``MAX_DT`` limits the assumed
# time between updates so sporadic frame stalls do not cause runaway predictions.
FACE_FILTER_ALPHA = 0.65
FACE_FILTER_BETA = 0.35
FACE_FILTER_GAMMA = 0.1
FACE_FILTER_MAX_DT = 1.0 / 15.0


def _validate_cascade(cascade: cv2.CascadeClassifier, name: str) -> cv2.CascadeClassifier:
    """Ensure that OpenCV successfully loaded a cascade classifier."""

    if cascade.empty():
        raise RuntimeError(
            "Unable to load the OpenCV {name} cascade. Ensure OpenCV data files "
            "are installed.".format(name=name)
        )
    return cascade


FACE_CASCADE = _validate_cascade(FACE_CASCADE, "frontal face")
EYE_CASCADE = _validate_cascade(EYE_CASCADE, "eye")


FaceRect = Tuple[int, int, int, int]


class AlphaBetaGammaFilter:
    """Smooth a series of 2D measurements while modelling constant acceleration."""

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._position: Optional[Tuple[float, float]] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._acceleration: Tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        """Clear the filter's state so the next update re-initialises it."""

        self._position = None
        self._velocity = (0.0, 0.0)
        self._acceleration = (0.0, 0.0)

    def update(self, measurement: Tuple[int, int], dt: float) -> Tuple[float, float]:
        """Incorporate ``measurement`` and return the filtered position."""

        if dt <= 0:
            dt = 0.0
        dt = min(dt, FACE_FILTER_MAX_DT)

        if self._position is None or dt == 0.0:
            self._position = (float(measurement[0]), float(measurement[1]))
            self._velocity = (0.0, 0.0)
            self._acceleration = (0.0, 0.0)
            return self._position

        px, py = self._position
        vx, vy = self._velocity
        ax, ay = self._acceleration

        # Predict where the subject should be given the current motion state.
        pred_x = px + vx * dt + 0.5 * ax * dt * dt
        pred_y = py + vy * dt + 0.5 * ay * dt * dt
        pred_vx = vx + ax * dt
        pred_vy = vy + ay * dt

        residual_x = float(measurement[0]) - pred_x
        residual_y = float(measurement[1]) - pred_y

        px = pred_x + self._alpha * residual_x
        py = pred_y + self._alpha * residual_y
        vx = pred_vx + (self._beta / max(dt, 1e-6)) * residual_x
        vy = pred_vy + (self._beta / max(dt, 1e-6)) * residual_y
        ax = ax + (2.0 * self._gamma / max(dt * dt, 1e-6)) * residual_x
        ay = ay + (2.0 * self._gamma / max(dt * dt, 1e-6)) * residual_y

        self._position = (px, py)
        self._velocity = (vx, vy)
        self._acceleration = (ax, ay)
        return self._position


def _split_stereo_faces(faces: Sequence[FaceRect], frame_width: int) -> Tuple[list[FaceRect], list[FaceRect]]:
    """Separate detections into left and right halves of a stereo frame."""

    half_width = frame_width // 2
    left_faces: list[FaceRect] = []
    right_faces: list[FaceRect] = []

    for (x, y, w, h) in faces:
        center_x = x + w / 2
        if center_x < half_width:
            left_faces.append((x, y, w, h))
        else:
            right_faces.append((x, y, w, h))

    return left_faces, right_faces


def _match_stereo_faces(
    faces: Sequence[FaceRect], frame_width: int, frame_height: int
) -> Optional[Tuple[FaceRect, FaceRect]]:
    """Return a pair of faces that likely depict the same person in stereo views."""

    left_faces, right_faces = _split_stereo_faces(faces, frame_width)
    if not left_faces or not right_faces:
        return None

    half_width = frame_width / 2.0
    best_score = float("inf")
    best_pair: Optional[Tuple[FaceRect, FaceRect]] = None

    for left in left_faces:
        lx, ly, lw, lh = left
        left_center_x = (lx + lw / 2.0) / half_width
        left_center_y = (ly + lh / 2.0) / frame_height
        left_width = lw / half_width

        for right in right_faces:
            rx, ry, rw, rh = right
            right_center_x = ((rx - half_width) + rw / 2.0) / half_width
            right_center_y = (ry + rh / 2.0) / frame_height
            right_width = rw / half_width

            dx = left_center_x - right_center_x
            dy = left_center_y - right_center_y
            dw = left_width - right_width
            score = dx * dx + dy * dy + 0.25 * dw * dw

            if score < best_score:
                best_score = score
                best_pair = (left, right)

    # Require strong similarity between the two detections.
    if best_score > 0.3 * 0.3:
        return None

    return best_pair


def _estimate_eye_midpoint(face: FaceRect, gray_frame) -> Tuple[int, int]:
    """Estimate the midpoint between the subject's eyes in ``face``."""

    x, y, w, h = face
    face_region = gray_frame[y : y + h, x : x + w]
    eyes = EYE_CASCADE.detectMultiScale(
        face_region, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
    )

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda bbox: bbox[2] * bbox[3], reverse=True)[:2]
        centers = [
            (x + ex + ew / 2.0, y + ey + eh / 2.0)
            for (ex, ey, ew, eh) in eyes
        ]
        mid_x = sum(center[0] for center in centers) / len(centers)
        mid_y = sum(center[1] for center in centers) / len(centers)
    elif len(eyes) == 1:
        ex, ey, ew, eh = max(eyes, key=lambda bbox: bbox[2] * bbox[3])
        mid_x = x + w / 2.0
        mid_y = y + ey + eh / 2.0
    else:
        mid_x = x + w / 2.0
        mid_y = y + h / 3.0

    return int(round(mid_x)), int(round(mid_y))


def _draw_cross(frame, face: FaceRect, midpoint: Tuple[int, int]) -> None:
    """Draw a crosshair centered on ``midpoint`` scaled relative to ``face``."""

    x, y, w, h = face
    center_x, center_y = midpoint
    half_length = max(int(min(w, h) * 0.15), 10)

    cv2.line(
        frame,
        (center_x - half_length, center_y),
        (center_x + half_length, center_y),
        (0, 0, 255),
        2,
    )
    cv2.line(
        frame,
        (center_x, center_y - half_length),
        (center_x, center_y + half_length),
        (0, 0, 255),
        2,
    )


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
    left_filter = AlphaBetaGammaFilter(
        FACE_FILTER_ALPHA, FACE_FILTER_BETA, FACE_FILTER_GAMMA
    )
    right_filter = AlphaBetaGammaFilter(
        FACE_FILTER_ALPHA, FACE_FILTER_BETA, FACE_FILTER_GAMMA
    )
    last_time = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Failed to read frame from the webcam.")

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            match = _match_stereo_faces(faces, frame.shape[1], frame.shape[0])

            if match is not None:
                left_face, right_face = match
                for face, tracker in (
                    (left_face, left_filter),
                    (right_face, right_filter),
                ):
                    midpoint = _estimate_eye_midpoint(face, gray)
                    filtered = tracker.update(midpoint, dt)
                    filtered_point = (int(round(filtered[0])), int(round(filtered[1])))
                    _draw_cross(frame, face, filtered_point)
            else:
                left_filter.reset()
                right_filter.reset()

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
